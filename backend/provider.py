import os
import requests
import litellm
import logging

logger = logging.getLogger(__name__)

class ProviderSettings:
    def __init__(self, provider_type="ollama", model_name="gemma4:12b-it-qat", api_base="http://localhost:11434", api_key=""):
        self.provider_type = provider_type  # "ollama", "xinference", "openai"
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key

class VLMSettings:
    def __init__(self, use_vlm=False, provider_type="ollama", model_name="gemma4:12b-it-qat", api_base="http://localhost:11434", api_key=""):
        self.use_vlm = use_vlm
        self.provider_type = provider_type
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key

# Global configurations, defaults to Ollama with the requested model
active_chat_provider = ProviderSettings()
active_vlm_provider = VLMSettings()

def get_active_provider():
    return active_chat_provider

def set_active_provider(provider_type, model_name, api_base=None, api_key=None):
    global active_chat_provider
    if provider_type == "ollama":
        api_base = api_base or "http://localhost:11434"
        api_key = api_key or "empty"
    elif provider_type == "xinference":
        api_base = api_base or "http://localhost:9997"
        api_key = api_key or "empty"
    elif provider_type == "openai":
        api_base = api_base or "https://api.openai.com/v1"
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}")
        
    active_chat_provider = ProviderSettings(provider_type, model_name, api_base, api_key)
    logger.info(f"Chat LLM Provider updated: {provider_type} ({model_name}) at {api_base}")
    return active_chat_provider

def get_vlm_provider():
    return active_vlm_provider

def set_vlm_provider(use_vlm, provider_type, model_name, api_base=None, api_key=None):
    global active_vlm_provider
    if provider_type == "ollama":
        api_base = api_base or "http://localhost:11434"
        api_key = api_key or "empty"
    elif provider_type == "xinference":
        api_base = api_base or "http://localhost:9997"
        api_key = api_key or "empty"
    elif provider_type == "openai":
        api_base = api_base or "https://api.openai.com/v1"
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    else:
        raise ValueError(f"Unknown VLM provider: {provider_type}")
        
    active_vlm_provider = VLMSettings(use_vlm, provider_type, model_name, api_base, api_key)
    logger.info(f"VLM Parser Provider updated: {provider_type} ({model_name}) at {api_base} (use_vlm={use_vlm})")
    return active_vlm_provider

# Status Checks
def check_ollama_status(api_base="http://localhost:11434"):
    try:
        url = f"{api_base.rstrip('/')}/api/tags"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"status": "online", "models": models}
    except Exception as e:
        logger.debug(f"Ollama offline check failed: {e}")
    return {"status": "offline", "models": []}

def check_xinference_status(api_base="http://localhost:9997"):
    try:
        # Check OpenAI-compatible endpoint
        url = f"{api_base.rstrip('/')}/v1/models"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"status": "online", "models": models}
    except Exception as e:
        logger.debug(f"Xinference check failed: {e}")
    
    # Try alternative API port
    try:
        url = f"{api_base.rstrip('/')}/api/v1/models"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            models = list(data.keys())
            return {"status": "online", "models": models}
    except Exception as e:
        logger.debug(f"Xinference API model check failed: {e}")
        
    return {"status": "offline", "models": []}

# Litellm patching logic
original_completion = litellm.completion
original_acompletion = litellm.acompletion

def _map_litellm_args(kwargs):
    global active_chat_provider, active_vlm_provider
    model = kwargs.get('model')
    
    # Clean the model string (e.g. remove litellm/ prefix if added by PageIndex client)
    if model:
        model = model.removeprefix("litellm/")
        
    # Decide which provider settings to apply based on model name
    is_vlm = False
    if active_vlm_provider.use_vlm and model:
        clean_model = model
        for prefix in ["openai/", "ollama/", "xinference/"]:
            if clean_model.startswith(prefix):
                clean_model = clean_model[len(prefix):]
                break
        clean_vlm_model = active_vlm_provider.model_name
        for prefix in ["openai/", "ollama/", "xinference/"]:
            if clean_vlm_model.startswith(prefix):
                clean_vlm_model = clean_vlm_model[len(prefix):]
                break
        if clean_model == clean_vlm_model:
            is_vlm = True

    if is_vlm:
        provider = active_vlm_provider
    else:
        provider = active_chat_provider
        
    if not model:
        model = provider.model_name
        
    if provider.provider_type == "ollama":
        if not model.startswith("ollama/"):
            model = f"ollama/{model}"
        kwargs['model'] = model
        if 'api_base' not in kwargs:
            kwargs['api_base'] = provider.api_base
        if 'api_key' not in kwargs:
            kwargs['api_key'] = provider.api_key or "empty"
    elif provider.provider_type == "xinference":
        # Extract model UID
        model_uid = model.split("/")[-1]
        kwargs['model'] = f"openai/{model_uid}"
        # Ensure base URL points to v1 endpoint
        api_base = provider.api_base
        if not api_base.endswith("/v1") and not api_base.endswith("/v1/"):
            api_base = f"{api_base.rstrip('/')}/v1"
        if 'api_base' not in kwargs:
            kwargs['api_base'] = api_base
        if 'api_key' not in kwargs:
            kwargs['api_key'] = provider.api_key or "empty"
    elif provider.provider_type == "openai":
        if not model.startswith("openai/"):
            model = f"openai/{model}"
        kwargs['model'] = model
        if 'api_base' not in kwargs and provider.api_base:
            kwargs['api_base'] = provider.api_base
        if 'api_key' not in kwargs and provider.api_key:
            kwargs['api_key'] = provider.api_key
            
    # Inject standard temperature if missing or 0 to keep it consistent
    if 'temperature' not in kwargs:
        kwargs['temperature'] = 0
        
    return kwargs

def patched_completion(*args, **kwargs):
    kwargs = _map_litellm_args(kwargs)
    return original_completion(*args, **kwargs)

async def patched_acompletion(*args, **kwargs):
    kwargs = _map_litellm_args(kwargs)
    return await original_acompletion(*args, **kwargs)

# Hook into litellm
litellm.completion = patched_completion
litellm.acompletion = patched_acompletion
litellm.drop_params = True
