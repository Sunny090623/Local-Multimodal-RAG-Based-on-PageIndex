import os
import sys
import json
import logging
import litellm
import asyncio
from pathlib import Path

# Ensure backend modules and PageIndex package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "PageIndex"))

from backend.provider import get_active_provider
from backend.shared import get_shared_client as get_document_client
from pageindex.utils import extract_json, remove_fields

logger = logging.getLogger(__name__)

async def run_llm_query(prompt, system_prompt=None):
    """Utility to run standard completions using active patched litellm."""
    provider = get_active_provider()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = await litellm.acompletion(
        model=provider.model_name,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

async def run_llm_query_stream(prompt, system_prompt=None):
    """Utility to stream completions using active patched litellm."""
    provider = get_active_provider()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = await litellm.acompletion(
            model=provider.model_name,
            messages=messages,
            temperature=0,
            stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Streaming LLM call failed: {e}", exc_info=True)
        yield f"\n[Generation Error: {e}]"

async def route_query_to_pages(doc_id, query, doc_type, structure):
    """Step 1: Read outline tree and use LLM to decide relevant pages/ranges."""
    structure_no_text = remove_fields(structure, fields=['text', 'summary', 'prefix_summary'])
    structure_json = json.dumps(structure_no_text, ensure_ascii=False, indent=2)
    
    prompt = f"""You are a document routing assistant. You are given a user query and a document's hierarchical outline tree.
Your job is to read the outline tree and identify the exact page numbers (for PDF documents) or starting line numbers (for Markdown/Text documents) that contain the information needed to answer the query.

Document Outline Tree:
{structure_json}

User Query:
{query}

Document Type: {"PDF" if doc_type == "pdf" else "Markdown/Text"}

Instructions:
1. Identify the most relevant sections.
2. Extract the page ranges (for PDF) or starting line numbers (for Markdown/Text) from the matching nodes.
3. Return a tight comma-separated range string. For example: "5-7" for pages 5 to 7, "3,8" for page 3 and 8, or "12" for page 12.
4. If the query cannot be answered by this document outline tree at all, return "none".

Response Format (return ONLY valid JSON, no markdown block, no explanation):
{{
    "thinking": "Brief explanation of which sections are relevant",
    "pages_string": "comma-separated ranges or single values, e.g., '3-5' or '12,14-16' or 'none'"
}}
"""
    try:
        raw_response = await run_llm_query(prompt)
        res_json = extract_json(raw_response)
        pages_str = res_json.get("pages_string", "none").strip()
        logger.info(f"LLM routed query to pages: {pages_str}")
        return pages_str
    except Exception as e:
        logger.error(f"Error routing query to pages: {e}", exc_info=True)
        return "none"

async def check_sufficiency_and_answer(query, context, pages_str):
    """Step 2: Check context sufficiency and attempt to answer the user query."""
    prompt = f"""You are a context checking and Q&A assistant.
Check if the provided document context contains enough information to fully and accurately answer the user query.

Document Context (from pages {pages_str}):
{context}

User Query:
{query}

Instructions:
1. Decide if the context is sufficient to answer the query. If it is only partially covered, or not mentioned at all, mark sufficient as false.
2. If sufficient, provide the complete answer based ONLY on the context.
3. If insufficient, output optimized keywords/query to search the web for the answer.

Response Format (return ONLY valid JSON, no markdown block, no explanation):
{{
    "sufficient": true or false,
    "thinking": "Brief explanation of sufficiency check",
    "answer": "Your answer if sufficient, otherwise leave empty",
    "search_query": "Search query keywords for web search if sufficient is false"
}}
"""
    try:
        raw_response = await run_llm_query(prompt)
        res_json = extract_json(raw_response)
        return res_json
    except Exception as e:
        logger.error(f"Error checking sufficiency: {e}", exc_info=True)
        return {"sufficient": False, "search_query": query, "thinking": "Error during sufficiency check, fallback to search"}

def get_document_fallback_context(doc_id: str, client) -> str:
    """Retrieves a fallback context from the document (either full text or node summaries)."""
    doc_info = client.documents.get(doc_id)
    if not doc_info:
        return ""
    
    # For text/markdown, try reading the original file directly
    doc_type = doc_info.get("type")
    path = doc_info.get("path")
    if doc_type in ["md", "txt"] and path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Fallback reading original file failed: {e}")
            
    # For PDF/Images, ensure doc is loaded in client
    client._ensure_doc_loaded(doc_id)
    
    # 1. Try to read all pages if total page count is small (e.g. <= 5 pages)
    page_count = doc_info.get("page_count", 0)
    if page_count > 0 and page_count <= 5:
        try:
            pages_str = ",".join(str(i) for i in range(1, page_count + 1))
            content_json_str = client.get_page_content(doc_id, pages_str)
            content_list = json.loads(content_json_str)
            context_str = ""
            for c in content_list:
                context_str += f"--- Page {c['page']} ---\n{c['content']}\n\n"
            return context_str
        except Exception as e:
            logger.error(f"Fallback context read failed: {e}")
            
    # 2. Otherwise, concatenate all node summaries in the structure tree
    structure = doc_info.get("structure", [])
    summaries = []
    
    def collect_summaries(nodes):
        for node in nodes:
            title = node.get("title", "")
            summary = node.get("summary", "")
            if title or summary:
                summaries.append(f"Section: {title}\nSummary: {summary}\n")
            if node.get("nodes"):
                collect_summaries(node["nodes"])
                
    collect_summaries(structure)
    return "\n".join(summaries)

# Generator-based Streaming RAG Flow
async def execute_rag_flow_stream(doc_ids, query, force_search=False):
    """Streams status updates and generated response tokens across multiple documents."""
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]
        
    logger.info(f"RAG query: {query!r} for doc_ids: {doc_ids!r}")
    client = get_document_client()
    
    valid_docs = []
    for d in doc_ids:
        doc_info = client.documents.get(d)
        if doc_info:
            valid_docs.append((d, doc_info))
            
    if not valid_docs:
        yield json.dumps({"type": "error", "content": f"Selected documents {doc_ids} not found."}) + "\n"
        return
        
    pages_inspected = [] # Will contain dict items: {"page": p, "doc_name": doc_name, "doc_type": doc_type}
    combined_context = ""
    fallback_active = False
    matched_any = False
    
    yield json.dumps({"type": "status", "content": "Analyzing structure trees for selected documents..."}) + "\n"
    
    for doc_id, doc_info in valid_docs:
        client._ensure_doc_loaded(doc_id)
        structure = doc_info.get("structure", [])
        doc_type = doc_info.get("type", "pdf")
        doc_name = doc_info.get("doc_name", "document")
        
        pages_str = await route_query_to_pages(doc_id, query, doc_type, structure)
        if pages_str != "none" and pages_str:
            yield json.dumps({"type": "status", "content": f"Outline matches in {doc_name} at: pages/lines {pages_str}. Retrieving context..."}) + "\n"
            try:
                content_json_str = client.get_page_content(doc_id, pages_str)
                logger.info(f"Retrieved content for {doc_name} pages {pages_str}: {content_json_str[:200]}...")
                content_list = json.loads(content_json_str)
                
                if "error" not in content_list:
                    doc_context = ""
                    for c in content_list:
                        doc_context += f"--- {doc_name} Page/Line {c['page']} ---\n{c['content']}\n\n"
                        pages_inspected.append({
                            "page": c['page'],
                            "doc_name": doc_name,
                            "doc_type": doc_type
                        })
                    combined_context += doc_context
                    matched_any = True
            except Exception as e:
                logger.error(f"Error fetching page content for {doc_name}: {e}")
                
    if matched_any and combined_context:
        yield json.dumps({"type": "status", "content": "Evaluating context sufficiency across selected documents..."}) + "\n"
        check_res = await check_sufficiency_and_answer(query, combined_context, "selected pages")
        logger.info(f"Sufficiency check result: {check_res}")
        
        if check_res.get("sufficient") is True:
            yield json.dumps({"type": "status", "content": "Context is sufficient. Stream-answering..."}) + "\n"
            
            prompt = f"""Answer the user query using ONLY the provided document contexts. Cite specific document names and page/line numbers in your answer.
            
Document Contexts:
{combined_context}

User Query:
{query}
"""
            async for token in run_llm_query_stream(prompt):
                yield json.dumps({"type": "delta", "content": token}) + "\n"
                
            yield json.dumps({"type": "result", "answer": "", "sources": pages_inspected, "fallback": False, "pages_inspected": pages_inspected}) + "\n"
            return
        else:
            fallback_active = True
            yield json.dumps({"type": "status", "content": "Selected contexts insufficient. Preparing to generate grounded response from all selected documents..."}) + "\n"
    else:
         fallback_active = True
         yield json.dumps({"type": "status", "content": "Query not matching document outline. Preparing to generate grounded response from all selected documents..."}) + "\n"
         
    if fallback_active:
        # Ground in fallback context of all selected documents
        combined_fallback = ""
        for doc_id, doc_info in valid_docs:
            doc_name = doc_info.get("doc_name", "document")
            fallback_ctx = get_document_fallback_context(doc_id, client)
            if fallback_ctx:
                combined_fallback += f"=== Content from {doc_name} ===\n{fallback_ctx}\n\n"
                
        logger.info(f"Fallback context retrieved: {len(combined_fallback)} characters.")
        yield json.dumps({"type": "status", "content": "Generating grounded answer from selected documents..."}) + "\n"
        
        prompt = f"""You are a helpful assistant. Answer the user query using the provided document context as the primary and highest priority source of truth.
If the query is a general greeting or conversational message, respond naturally.
Otherwise, answer based strictly on the document context. If the context does not contain the information needed to answer the query, state that the documents do not mention it.

Document Context:
{combined_fallback}

User Query:
{query}
"""
        async for token in run_llm_query_stream(prompt):
            yield json.dumps({"type": "delta", "content": token}) + "\n"
            
        yield json.dumps({"type": "result", "answer": "", "sources": [], "fallback": True, "pages_inspected": pages_inspected}) + "\n"
        return
