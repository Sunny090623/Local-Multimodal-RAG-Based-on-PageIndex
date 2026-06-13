import os
import sys
import json
import logging
import litellm
import asyncio
from pathlib import Path
from duckduckgo_search import DDGS

# Ensure backend modules and PageIndex package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "PageIndex"))

from backend.provider import get_active_provider
from backend.shared import get_shared_client as get_document_client
from pageindex.utils import extract_json, remove_fields

logger = logging.getLogger(__name__)

# Global switch to enable/disable web search fallback.
# Set to False per user request to avoid blocking errors/local network issues.
WEB_SEARCH_ENABLED = False

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

async def execute_web_search(search_query):
    """Executes DuckDuckGo search with strict exception handling and rate-limiting safety."""
    search_results = []
    error_msg = None
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=5)
            if results:
                for r in results:
                    search_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("url", "")),
                        "snippet": r.get("body", "")
                    })
    except Exception as e:
        err_str = str(e)
        logger.error(f"DuckDuckGo search fallback failed: {err_str}")
        error_msg = err_str
        
    return search_results, error_msg

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
async def execute_rag_flow_stream(doc_id, query, force_search=False):
    """Streams status updates and generated response tokens."""
    logger.info(f"RAG query: {query!r} for doc_id: {doc_id!r} (force_search={force_search})")
    client = get_document_client()
    doc_info = client.documents.get(doc_id)
    if not doc_info:
        yield json.dumps({"type": "error", "content": f"Document {doc_id} not found."}) + "\n"
        return
        
    client._ensure_doc_loaded(doc_id)
    structure = doc_info.get("structure", [])
    doc_type = doc_info.get("type", "pdf")
    doc_name = doc_info.get("doc_name", "document")
    
    pages_inspected = []
    fallback_active = False
    search_results = []
    err = None
    
    # Check if forced search is on
    if force_search:
        if not WEB_SEARCH_ENABLED:
            yield json.dumps({"type": "status", "content": "Web Search is currently disabled. Generating direct response using LLM..."}) + "\n"
            prompt = f"""Answer the user query based on your general knowledge.

User Query:
{query}
"""
            async for token in run_llm_query_stream(prompt):
                yield json.dumps({"type": "delta", "content": token}) + "\n"
            yield json.dumps({"type": "result", "answer": "", "sources": [], "fallback": True, "pages_inspected": []}) + "\n"
            return

        fallback_active = True
        yield json.dumps({"type": "status", "content": "Web Search forced by settings. Executing search..."}) + "\n"
        search_results, err = await execute_web_search(query)
        
        if err:
            logger.error(f"Forced search failed: {err}")
            is_rate_lim = "403" in err or "Forbidden" in err or "rate" in err.lower()
            err_msg = "Web search fallback failed: Local search blocked (rate-limited), please check connection." if is_rate_lim else f"Web search fallback failed: Local search blocked, please check connection. (Error: {err})"
            yield json.dumps({"type": "error", "content": err_msg}) + "\n"
            yield json.dumps({"type": "result", "answer": err_msg, "sources": [], "fallback": True, "pages_inspected": []}) + "\n"
            return
            
        yield json.dumps({"type": "status", "content": f"Found {len(search_results)} search results. Generating answer..."}) + "\n"
        
        context_str = ""
        sources = []
        for idx, r in enumerate(search_results, 1):
            context_str += f"[{idx}] Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n\n"
            sources.append({"id": idx, "title": r["title"], "url": r["url"]})
            
        prompt = f"""You are a web search grounding assistant. Answer the user query using the provided web search context.
Cite the source numbers in brackets (e.g. [1], [2]) corresponding to the index of search results when stating facts.

Web Search Context:
{context_str}

User Query:
{query}
"""
        async for token in run_llm_query_stream(prompt):
            yield json.dumps({"type": "delta", "content": token}) + "\n"
            
        yield json.dumps({"type": "result", "answer": "", "sources": sources, "fallback": True, "pages_inspected": []}) + "\n"
        return

    # Normal RAG search routing
    yield json.dumps({"type": "status", "content": "Analyzing document structure tree..."}) + "\n"
    pages_str = await route_query_to_pages(doc_id, query, doc_type, structure)
    
    context_str = ""
    if pages_str == "none" or not pages_str:
        fallback_active = True
        if not WEB_SEARCH_ENABLED:
            yield json.dumps({"type": "status", "content": "Query not matching document outline. Web search is disabled, preparing to generate document-priority response..."}) + "\n"
        else:
            yield json.dumps({"type": "status", "content": "Query not matching document outline. Triggering Web search fallback..."}) + "\n"
            search_results, err = await execute_web_search(query)
    else:
        yield json.dumps({"type": "status", "content": f"Outline matches found at: pages/lines {pages_str}. Retrieving context..."}) + "\n"
        try:
            content_json_str = client.get_page_content(doc_id, pages_str)
            logger.info(f"Retrieved page content for pages {pages_str}: {content_json_str[:200]}...")
            content_list = json.loads(content_json_str)
            
            if "error" in content_list:
                raise ValueError(content_list["error"])
                
            for c in content_list:
                context_str += f"--- Page {c['page']} ---\n{c['content']}\n\n"
                pages_inspected.append(c['page'])
        except Exception as e:
            logger.error(f"Error fetching page content: {e}")
            context_str = ""
            
        if not context_str:
            fallback_active = True
            if not WEB_SEARCH_ENABLED:
                yield json.dumps({"type": "status", "content": "Could not read page contents. Web search is disabled, preparing to generate document-priority response..."}) + "\n"
            else:
                yield json.dumps({"type": "status", "content": "Could not read page contents. Triggering Web search fallback..."}) + "\n"
                search_results, err = await execute_web_search(query)
        else:
            yield json.dumps({"type": "status", "content": "Evaluating context sufficiency..."}) + "\n"
            check_res = await check_sufficiency_and_answer(query, context_str, pages_str)
            logger.info(f"Sufficiency check result: {check_res}")
            
            if check_res.get("sufficient") is True:
                # Answer is sufficient, stream the pre-generated answer or stream a final generation using the context
                yield json.dumps({"type": "status", "content": "Document context is sufficient. Stream-answering..."}) + "\n"
                
                # Stream synthesis to make response lively
                prompt = f"""Answer the user query using ONLY the provided document context. Cite specific page numbers in your answer.
                
Document Context:
{context_str}

User Query:
{query}
"""
                async for token in run_llm_query_stream(prompt):
                    yield json.dumps({"type": "delta", "content": token}) + "\n"
                    
                sources = [{"page": p, "doc_name": doc_name} for p in pages_inspected]
                yield json.dumps({"type": "result", "answer": "", "sources": sources, "fallback": False, "pages_inspected": pages_inspected}) + "\n"
                return
            else:
                fallback_active = True
                search_q = check_res.get("search_query", query)
                if not WEB_SEARCH_ENABLED:
                    yield json.dumps({"type": "status", "content": "Document context insufficient. Web search is disabled, preparing to generate document-priority response..."}) + "\n"
                else:
                    yield json.dumps({"type": "status", "content": f"Document context insufficient. Triggering Web search for: '{search_q}'..."}) + "\n"
                    search_results, err = await execute_web_search(search_q)

    # If fallback is active, stream search response
    if fallback_active:
        if not WEB_SEARCH_ENABLED:
            # Generate fallback context from the document if available
            fallback_context = context_str
            if not fallback_context and doc_id:
                fallback_context = get_document_fallback_context(doc_id, client)
            logger.info(f"Fallback context retrieved: {len(fallback_context)} characters.")
                
            yield json.dumps({"type": "status", "content": "Generating grounded answer from document..."}) + "\n"
            
            prompt = f"""You are a helpful assistant. Answer the user query using the provided document context as the primary and highest priority source of truth.
If the query is a general greeting or conversational message, respond naturally.
Otherwise, answer based strictly on the document context. If the context does not contain the information needed to answer the query, state that the document does not mention it.

Document Context:
{fallback_context}

User Query:
{query}
"""
            async for token in run_llm_query_stream(prompt):
                yield json.dumps({"type": "delta", "content": token}) + "\n"
                
            yield json.dumps({"type": "result", "answer": "", "sources": [], "fallback": True, "pages_inspected": pages_inspected}) + "\n"
            return

        if err:
            logger.error(f"Web search fallback failed: {err}")
            is_rate_lim = "403" in err or "Forbidden" in err or "rate" in err.lower()
            err_msg = "Web search fallback failed: Local search blocked (rate-limited), please check connection." if is_rate_lim else f"Web search fallback failed: Local search blocked, please check connection. (Error: {err})"
            yield json.dumps({"type": "error", "content": err_msg}) + "\n"
            yield json.dumps({"type": "result", "answer": err_msg, "sources": [], "fallback": True, "pages_inspected": pages_inspected}) + "\n"
            return
            
        if not search_results:
            msg = "Document content is insufficient, and web search returned no results."
            yield json.dumps({"type": "error", "content": msg}) + "\n"
            yield json.dumps({"type": "result", "answer": msg, "sources": [], "fallback": True, "pages_inspected": pages_inspected}) + "\n"
            return
            
        yield json.dumps({"type": "status", "content": f"Found {len(search_results)} search results. Synthesizing final answer..."}) + "\n"
        
        context_str = ""
        sources = []
        for idx, r in enumerate(search_results, 1):
            context_str += f"[{idx}] Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n\n"
            sources.append({"id": idx, "title": r["title"], "url": r["url"]})
            
        prompt = f"""You are a web search grounding assistant. Answer the user query using the provided web search context.
Cite the source numbers in brackets (e.g. [1], [2]) corresponding to the index of search results when stating facts.

Web Search Context:
{context_str}

User Query:
{query}
"""
        async for token in run_llm_query_stream(prompt):
            yield json.dumps({"type": "delta", "content": token}) + "\n"
            
        yield json.dumps({"type": "result", "answer": "", "sources": sources, "fallback": True, "pages_inspected": pages_inspected}) + "\n"
