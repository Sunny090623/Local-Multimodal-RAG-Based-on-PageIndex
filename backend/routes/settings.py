"""
Settings routes — LLM provider status, Chat settings, and VLM settings.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from backend.provider import (
    set_active_provider, get_active_provider,
    check_ollama_status, check_xinference_status,
    get_vlm_provider, set_vlm_provider
)
from backend.shared import STORAGE_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])

# Settings file paths
SETTINGS_FILE = STORAGE_DIR / "settings.json"
SETTINGS_CHAT_FILE = STORAGE_DIR / "settings_chat.json"
SETTINGS_VLM_FILE = STORAGE_DIR / "settings_vlm.json"


# --- Pydantic models ---

class ChatSettingsPayload(BaseModel):
    provider_type: str
    model_name: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None

class VLMSettingsPayload(BaseModel):
    use_vlm: bool = False
    vlm_provider_type: Optional[str] = "ollama"
    vlm_model: Optional[str] = None
    vlm_api_base: Optional[str] = None
    vlm_api_key: Optional[str] = None


# --- Settings persistence ---

def load_settings():
    """Load Chat and VLM settings from disk, generating defaults if missing."""
    import os

    # If old settings.json exists, remove it to ensure the key is only in the two config files
    if SETTINGS_FILE.exists():
        try:
            os.remove(SETTINGS_FILE)
            logger.info("Old settings.json deleted to enforce single-place persistence of API key.")
        except Exception as e:
            logger.error(f"Failed to delete old settings.json: {e}")

    # Auto-generate settings_chat.json if missing
    if not SETTINGS_CHAT_FILE.exists():
        try:
            default_chat = {
                "provider_type": "ollama",
                "model_name": "gemma4:12b-it-qat",
                "api_base": "http://localhost:11434",
                "api_key": "empty"
            }
            with open(SETTINGS_CHAT_FILE, "w") as f:
                json.dump(default_chat, f, indent=2)
            logger.info("Automatically generated default settings_chat.json")
        except Exception as e:
            logger.error(f"Failed to generate default settings_chat.json: {e}")

    # Auto-generate settings_vlm.json if missing
    if not SETTINGS_VLM_FILE.exists():
        try:
            default_vlm = {
                "use_vlm": False,
                "vlm_provider_type": "ollama",
                "vlm_model": "gemma4:12b-it-qat",
                "vlm_api_base": "http://localhost:11434",
                "vlm_api_key": "empty"
            }
            with open(SETTINGS_VLM_FILE, "w") as f:
                json.dump(default_vlm, f, indent=2)
            logger.info("Automatically generated default settings_vlm.json")
        except Exception as e:
            logger.error(f"Failed to generate default settings_vlm.json: {e}")

    # Load Chat Settings
    chat_loaded = False
    if SETTINGS_CHAT_FILE.exists():
        try:
            with open(SETTINGS_CHAT_FILE, "r") as f:
                data = json.load(f)
                set_active_provider(
                    provider_type=data.get("provider_type", "ollama"),
                    model_name=data.get("model_name", "gemma4:12b-it-qat"),
                    api_base=data.get("api_base"),
                    api_key=data.get("api_key")
                )
                chat_loaded = True
                logger.info("Saved Chat settings loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load saved Chat settings: {e}")
            
    if not chat_loaded:
        set_active_provider("ollama", "gemma4:12b-it-qat", "http://localhost:11434")

    # Load VLM Settings
    vlm_loaded = False
    if SETTINGS_VLM_FILE.exists():
        try:
            with open(SETTINGS_VLM_FILE, "r") as f:
                data = json.load(f)
                set_vlm_provider(
                    use_vlm=True,
                    provider_type=data.get("vlm_provider_type") or "ollama",
                    model_name=data.get("vlm_model") or "gemma4:12b-it-qat",
                    api_base=data.get("vlm_api_base"),
                    api_key=data.get("vlm_api_key")
                )
                vlm_loaded = True
                logger.info("Saved VLM settings loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load saved VLM settings: {e}")
            
    if not vlm_loaded:
        set_vlm_provider(True, "ollama", "gemma4:12b-it-qat", "http://localhost:11434")


# --- Endpoints ---

@router.get("/status")
async def get_status(ollama_base: str = "http://localhost:11434", xinference_base: str = "http://localhost:9997"):
    # Run blocking HTTP status checks in thread pool to avoid blocking the event loop
    ollama_info, xinference_info = await asyncio.gather(
        asyncio.to_thread(check_ollama_status, ollama_base),
        asyncio.to_thread(check_xinference_status, xinference_base)
    )
    
    active = get_active_provider()
    vlm = get_vlm_provider()
    
    return {
        "ollama": ollama_info,
        "xinference": xinference_info,
        "active_provider": {
            "provider_type": active.provider_type,
            "model_name": active.model_name,
            "api_base": active.api_base,
            "api_key": "***" if active.api_key else ""
        },
        "vlm_provider": {
            "use_vlm": vlm.use_vlm,
            "provider_type": vlm.provider_type,
            "model_name": vlm.model_name,
            "api_base": vlm.api_base,
            "api_key": "***" if vlm.api_key else ""
        }
    }

@router.post("/settings/chat")
async def update_chat_settings(payload: ChatSettingsPayload):
    try:
        chat_key = payload.api_key
        if chat_key == "***":
            chat_key = get_active_provider().api_key
            
        set_active_provider(
            provider_type=payload.provider_type,
            model_name=payload.model_name,
            api_base=payload.api_base,
            api_key=chat_key
        )
        
        # Save to file
        settings_data = payload.dict()
        if settings_data.get("api_key") == "***":
            settings_data["api_key"] = get_active_provider().api_key
            
        with open(SETTINGS_CHAT_FILE, "w") as f:
            json.dump(settings_data, f, indent=2)
            
        return {"status": "success", "message": "Chat settings updated and saved."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/settings/vlm")
async def update_vlm_settings(payload: VLMSettingsPayload):
    try:
        vlm_key = payload.vlm_api_key
        if vlm_key == "***":
            vlm_key = get_vlm_provider().api_key
            
        set_vlm_provider(
            use_vlm=True,
            provider_type=payload.vlm_provider_type or "ollama",
            model_name=payload.vlm_model or "gemma4:12b-it-qat",
            api_base=payload.vlm_api_base,
            api_key=vlm_key
        )
        
        # Save to file
        settings_data = payload.dict()
        settings_data["use_vlm"] = True
        if settings_data.get("vlm_api_key") == "***":
            settings_data["vlm_api_key"] = get_vlm_provider().api_key
            
        with open(SETTINGS_VLM_FILE, "w") as f:
            json.dump(settings_data, f, indent=2)
            
        return {"status": "success", "message": "VLM settings updated and saved."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
