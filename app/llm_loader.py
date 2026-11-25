import os
import threading
from django.conf import settings

_llm = None
_lock = threading.Lock()


def get_llm():
    global _llm
    with _lock:
        if _llm is None:
            from llama_cpp import Llama
            _llm = Llama(
                model_path=str(settings.LLM_MODEL_PATH),
                n_ctx=4096,
                n_threads=max(2, (os.cpu_count() or 4) // 2),
                n_gpu_layers=0,
                verbose=False,
            )
        return _llm
