"""Model adapters. Import the stub by default; real adapters are optional deps.

    from toduq.runners import EchoClient
    from toduq.runners.claude import ClaudeClient        # needs `anthropic`
    from toduq.runners.openai import OpenAIClient         # needs `openai`
    from toduq.runners.open_source import VLLMClient      # needs `vllm`/`transformers`
"""
from toduq.runners.base import EchoClient, GenConfig, LLMClient
from toduq.runners.pool import ModelPool

__all__ = ["EchoClient", "GenConfig", "LLMClient", "ModelPool"]
