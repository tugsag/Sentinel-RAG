import ollama

# 1. Capture the original chat method
original_chat = ollama.AsyncClient.chat

# 2. Define a wrapper that "cleans" the arguments
async def patched_chat(self, *args, **kwargs):
    # These are the args that Ollama 2026 actually accepts
    valid_keys = {"model", "messages", "format", "options", "stream", "keep_alive", "tools"}
    
    # Move 'temperature', 'top_p', etc. into the 'options' dict if they are loose
    if "options" not in kwargs:
        kwargs["options"] = {}
    
    # Pluck out common loose parameters and move them
    for key in ["temperature", "top_p", "seed", "num_ctx"]:
        if key in kwargs:
            kwargs["options"][key] = kwargs.pop(key)

    # Remove any other non-standard args that NeMo might be injecting
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    return await original_chat(self, *args, **filtered_kwargs)

# 3. Apply the patch
ollama.AsyncClient.chat = patched_chat
print('patch applied')