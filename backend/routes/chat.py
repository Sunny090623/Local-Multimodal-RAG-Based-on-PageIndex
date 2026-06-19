"""
Chat routes — RAG chat streaming endpoint.
"""

import logging
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.rag import execute_rag_flow_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatPayload(BaseModel):
    doc_ids: Optional[List[str]] = None
    doc_id: Optional[str] = None
    query: str
    force_search: bool = False


@router.post("/chat")
async def chat_query(payload: ChatPayload):
    doc_ids = payload.doc_ids
    if not doc_ids and payload.doc_id:
        doc_ids = [payload.doc_id]
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No documents selected for chat.")
    generator = execute_rag_flow_stream(doc_ids, payload.query, payload.force_search)
    return StreamingResponse(generator, media_type="text/event-stream")
