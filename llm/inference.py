"""
Real (lightweight) LLM inference.

We use `google/flan-t5-small` — a ~80MB instruction-tuned model that:
  - Runs on CPU (no GPU required for development)
  - Actually answers questions sensibly (unlike base GPT-2)
  - Is small enough that 1000 concurrent requests are still tractable
    on a laptop while we study the *distributed* parts of the system.

Design notes:
  - The model is loaded LAZILY (on first call) so `import` stays fast.
  - We load it ONCE globally and share it across all workers via a lock,
    because loading takes a few seconds and uses ~300MB of RAM.
    In a real distributed system each GPU node would load its own copy;
    sharing-with-a-lock is a fair simulation for a single-machine demo.
  - Inference is wrapped in a threading.Lock because PyTorch model
    forward passes from many threads at once can corrupt internal state.
"""

import threading
import time

# We import the heavy libs *inside* the lazy loader so importing this
# module is cheap even when the model is never used (e.g. during tests).
_MODEL_NAME = "google/flan-t5-small"
_tokenizer = None
_model = None
_load_lock = threading.Lock()      # protects the one-time model load
_infer_lock = threading.Lock()     # protects each generate() call


def _ensure_loaded():
    """Load the model exactly once, the first time it's needed."""
    global _tokenizer, _model
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:        # double-checked locking
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as e:
            raise RuntimeError(
                "transformers is not installed. Run:\n"
                "  pip install transformers torch sentencepiece"
            ) from e
        print(f"[LLM] Loading {_MODEL_NAME} (first call only)...")
        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
        _model.eval()                  # turn off dropout, training-only ops
        print(f"[LLM] Loaded in {time.time() - t0:.1f}s")


def run_llm(query: str, context: str, max_new_tokens: int = 80) -> str:
    """Generate an answer to `query` grounded in `context`.

    Args:
        query: the user's question (from RAG-augmented Request).
        context: retrieved passages from the RAG retriever.
        max_new_tokens: cap on output length (kept small for speed).

    Returns:
        The model's answer as a plain string.
    """
    _ensure_loaded()

    # FLAN-T5 is instruction-tuned: just describe the task in the prompt.
    prompt = (
        "Answer the question using the context.\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        "Answer:"
    )

    # Tokenize -> generate -> decode. We hold _infer_lock so concurrent
    # workers don't trample each other's GPU/CPU state.
    with _infer_lock:
        import torch
        inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():            # no gradients needed for inference
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,         # deterministic = reproducible demos
            )
        answer = _tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return answer.strip()
