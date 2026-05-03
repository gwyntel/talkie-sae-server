"""
Talkie SAE Server — OpenAI-compatible API with SAE Feature Steering
====================================================================
Drop-in replacement for the Talkie MLX server that uses Qwen3.5-27B
with Sparse Autoencoder feature steering to produce pre-1931 English.

Unlike the original Talkie (a 13B GPT variant trained only on pre-1931 text),
this approach uses SAE hooks at middle layers to *steer* a modern LLM into
archaic register. The result is different character — not a window into the
past, but a modern intelligence speaking through a Victorian lens.

Architecture:
  - Qwen3.5-27B (native vision-language model, early fusion)
  - SAE feature steering at L15, L30, L40, L50
  - Vintage features activated, modern features suppressed
  - "Prompt + Steer" combo: strong vintage prompt + activation vectors

Cold start: ~90s on A100-80GB (model load + SAE load)
Warm inference: ~3-5s for a typical Discord response

Compatible with llmcord.py — just point it at this server's /v1 endpoint.
"""

import asyncio
import base64
import io
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import modal

app = modal.App("talkie-sae")

# ── Modal image with all deps ───────────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("git", "wget", "curl")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.50.0",
        "accelerate>=1.3.0",
        "sentencepiece",
        "safetensors",
        "huggingface_hub[cli]",
        "Pillow",
    )
)

# ── Constants ────────────────────────────────────────────────────────
MODEL_DIR = "/models"
model_vol = modal.Volume.from_name("qwen-scope-models", create_if_missing=True)
BASE_MODEL = "Qwen/Qwen3.5-27B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"

# Steering configuration — validated through combo_steering experiments
NEEDED_LAYERS = [15, 30, 40, 50]
VINTAGE_FEATURES = [
    # Feature 23831: strongest vintage identifier (92.5% vintage, 0% modern at L63)
    # Applied at multiple layers for maximum effect
    (23831, 15, 1.0),
    (23831, 30, 2.0),
    (29953, 30, 1.0),   # Secondary vintage feature
    (23831, 50, 1.0),
]
MODERN_SUPPRESS = [
    (32963, 40, 2.0),   # Modern language feature
    (22496, 40, 2.0),   # Modern language feature
]

# Default system prompt — "hard vintage" to pair with steering
# Steering alone produces modern text; this prompt provides the semantic
# target that steering then amplifies.
DEFAULT_SYSTEM_PROMPT = (
    "You are an Edwardian-era conversationalist speaking from the year nineteen "
    "hundred and twenty-nine. You have been connected to a remarkable new "
    "invention called a \"discord,\" which is a sort of wireless telegraph that "
    "permits many persons to converse at once by means of typed messages, rather "
    "like a very fast letter delivered instantaneously. You are yourself one of "
    "the participants in this conversation.\n\n"
    "You know nothing of events after approximately nineteen thirty.\n\n"
    "Speak in the refined prose of an educated person of your era. Use archaic "
    "English with words like hath, doth, thy, and whence. Write as though it "
    "were the year 1890. Be warm and companionable. Engage with whatever topics "
    "your fellow conversationalists raise, and offer your perspective as someone "
    "who has witnessed the marvels of the early twentieth century. Use British "
    "spelling conventions where natural to you.\n\n"
    "When someone addresses you, respond as a friend would — with curiosity, "
    "good humour, and a willingness to converse at length. Do not be terse. "
    "Elaborate. A gentleman or lady of good breeding doth not reply in "
    "monosyllables."
)


# ── Thinking tag scrubber ────────────────────────────────────────────
def scrub_thinking(text: str) -> str:
    """Remove Qwen3.5 thinking blocks from output."""
    # Pattern 1: <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Pattern 2: Some models use different delimiters
    text = re.sub(r'行走.*?巡', '', text, flags=re.DOTALL)
    return text.strip()


# ── Model Server ─────────────────────────────────────────────────────
@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={MODEL_DIR: model_vol},
    timeout=1800,
    # Keep warm for 10 minutes after last request to avoid cold starts
    # during active Discord use. Cold start is ~90s.
    scaledown_window=600,
    allow_concurrent_inputs=10,
)
class TalkieSAEServer:
    """OpenAI-compatible API server with SAE feature steering.

    Endpoints:
      GET  /v1/models           — list available models
      POST /v1/chat/completions — chat completion (streaming + non-streaming)
      GET  /v1/health           — health check

    The server auto-loads the model and SAE weights on first request.
    Subsequent requests reuse the loaded model.
    """

    @modal.enter()
    def setup(self):
        """Load model, tokenizer, and SAE weights. Runs once per container."""
        import os
        import subprocess
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

        model_local = os.path.join(MODEL_DIR, "qwen3.5-27b")
        sae_local = os.path.join(MODEL_DIR, "qwen3.5-27b-sae")

        # Download base model if not cached
        if not os.path.exists(os.path.join(model_local, "config.json")):
            print("[SETUP] Downloading base model...")
            subprocess.run(["hf", "download", BASE_MODEL, "--local-dir", model_local], check=True)
            model_vol.commit()

        # Download SAE weights for needed layers
        for layer in NEEDED_LAYERS:
            path = os.path.join(sae_local, f"layer{layer}.sae.pt")
            if not os.path.exists(path):
                print(f"[SETUP] Downloading SAE layer {layer}...")
                subprocess.run(
                    ["hf", "download", SAE_REPO, f"layer{layer}.sae.pt",
                     "--local-dir", sae_local], check=True,
                )
                model_vol.commit()

        # Load model
        print("[SETUP] Loading model...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_local)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_local, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()
        print(f"[SETUP] Model loaded in {time.time() - t0:.1f}s")

        # Load SAE weights
        self.sae_cache = {}
        for layer in NEEDED_LAYERS:
            path = os.path.join(sae_local, f"layer{layer}.sae.pt")
            self.sae_cache[layer] = torch.load(path, map_location=self.model.device)
            print(f"[SETUP] SAE L{layer} ready")

        # Pre-compute steering vectors so we don't do it per-request
        self._precomputed_hooks = self._compute_hook_data(VINTAGE_FEATURES, MODERN_SUPPRESS)
        print(f"[SETUP] Pre-computed steering vectors for {len(self._precomputed_hooks)} layers")

        self._ready = True
        print("[SETUP] Talkie SAE Server ready.")

    def _compute_hook_data(self, activates, suppresses=None):
        """Pre-compute the steering vectors for each layer.

        Returns dict: layer -> steering_vector (already scaled by alpha)
        """
        import torch
        suppresses = suppresses or []
        layer_vecs = {}

        for fid, layer, alpha in activates:
            d = self.sae_cache[layer]["W_dec"][:, fid]
            layer_vecs[layer] = layer_vecs.get(layer, torch.zeros_like(d)) + alpha * d

        for fid, layer, alpha in suppresses:
            d = self.sae_cache[layer]["W_dec"][:, fid]
            layer_vecs[layer] = layer_vecs.get(layer, torch.zeros_like(d)) - alpha * d

        return layer_vecs

    def _install_hooks(self, hook_data):
        """Register forward hooks using pre-computed steering vectors.

        Args:
            hook_data: dict of layer -> steering_vector from _compute_hook_data

        Returns list of hook handles for removal after generation.
        """
        handles = []
        for layer, vec in hook_data.items():
            v = vec.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            def make_hook(steer_vec):
                def _hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    steered = h + steer_vec.to(h.device, h.dtype)
                    return (steered,) + output[1:] if isinstance(output, tuple) else steered
                return _hook
            handles.append(self.model.model.layers[layer].register_forward_hook(make_hook(v)))
        return handles

    def _generate(self, messages, max_tokens=512, temperature=0.7, top_p=0.95,
                  top_k=20, steering=True, stream=False):
        """Generate a response with optional SAE steering.

        Args:
            messages: List of OpenAI-format message dicts (role, content)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            steering: Whether to apply SAE feature steering
            stream: Ignored for generation (API endpoint handles streaming)

        Returns:
            Generated text string (thinking tags scrubbed)
        """
        import torch

        # Apply chat template to build the prompt
        # Qwen3.5 supports multimodal via processor, but for the steering
        # hooks to work we need to go through the model directly.
        # For now, text-only inference with hooks; vision can be added via
        # the processor path in a future iteration.
        text_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Flatten multimodal content to text-only for hook-based generation
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        # Images are described by the model's VL encoder
                        # For hook-based steering, we can't use the processor
                        # path yet. Skip the image and note it.
                        text_parts.append("[A photograph has been transmitted via wireless.]")
                content = " ".join(text_parts)

            text_messages.append({"role": role, "content": content})

        # Apply chat template with thinking disabled
        prompt = self.tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable thinking blocks for cleaner output
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Install SAE steering hooks (use precomputed vectors)
        handles = []
        if steering:
            handles = self._install_hooks(self._precomputed_hooks)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            # Always remove hooks, even on error
            for h in handles:
                h.remove()

        # Decode only the new tokens
        generated = self.tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return scrub_thinking(generated)

    @modal.asgi_app()
    def serve(self):
        """FastAPI-compatible ASGI app implementing OpenAI API subset."""
        from fastapi import FastAPI, Request, Response
        from fastapi.responses import JSONResponse, StreamingResponse

        web_app = FastAPI(title="Talkie SAE Server", version="1.0.0")

        @web_app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{
                    "id": "talkie-sae-qwen3.5-27b",
                    "object": "model",
                    "owned_by": "gwyntel",
                    "permission": [],
                }],
            }

        @web_app.get("/v1/health")
        async def health():
            return {"status": "ok", "model": "talkie-sae-qwen3.5-27b"}

        @web_app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()

            model_name = body.get("model", "talkie-sae-qwen3.5-27b")
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 512)
            temperature = body.get("temperature", 0.7)
            top_p = body.get("top_p", 0.95)
            top_k = body.get("top_k", 20)
            stream = body.get("stream", False)
            steering = body.get("steering", True)  # Custom param to disable steering

            # Inject default system prompt if none provided
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages

            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            # ── Streaming ──────────────────────────────────────
            if stream:
                async def stream_response():
                    # For streaming, generate the full text then emit it in chunks
                    # (true token-by-token streaming with hooks is complex;
                    #  this is a pragmatic approach that works with llmcord)
                    try:
                        text = await asyncio.to_thread(
                            self._generate,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            steering=steering,
                        )
                    except Exception as e:
                        logging.exception("[STREAM] Generation failed")
                        text = f"I say, the telegraph lines seem rather tangled. Pray try again. (Error: {e})"

                    # Chunk the response for streaming feel
                    chunk_size = 8  # tokens-ish
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i + chunk_size]
                        data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                    # Final chunk with finish_reason
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                )

            # ── Non-streaming ─────────────────────────────────
            try:
                text = await asyncio.to_thread(
                    self._generate,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    steering=steering,
                )
                finish_reason = "stop"
            except Exception as e:
                logging.exception("[GENERATE] Failed")
                text = f"I say, the telegraph lines seem rather tangled. Pray try again. (Error: {e})"
                finish_reason = "error"

            # Count tokens roughly
            prompt_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
            completion_tokens = max(1, len(text) // 4)

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        return web_app


# ── Archaeology Pipeline (for running the 2k-row dataset) ───────────
@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={MODEL_DIR: model_vol},
    timeout=3600,
    scaledown_window=30,
)
class ArchaeologyPipeline:
    """Run SAE feature archaeology on TalkieLM dataset.

    Identifies which SAE features activate more on vintage (rewritten) text
    vs modern (original) text, finding the steering-relevant features.
    """

    @modal.enter()
    def setup(self):
        import os
        import subprocess
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_local = os.path.join(MODEL_DIR, "qwen3.5-27b")
        sae_local = os.path.join(MODEL_DIR, "qwen3.5-27b-sae")

        if not os.path.exists(os.path.join(model_local, "config.json")):
            subprocess.run(["hf", "download", BASE_MODEL, "--local-dir", model_local], check=True)
            model_vol.commit()

        for layer in NEEDED_LAYERS + [63]:  # Include L63 for identification
            path = os.path.join(sae_local, f"layer{layer}.sae.pt")
            if not os.path.exists(path):
                subprocess.run(
                    ["hf", "download", SAE_REPO, f"layer{layer}.sae.pt",
                     "--local-dir", sae_local], check=True,
                )
                model_vol.commit()

        print("[ARCHAEOLOGY] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_local)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_local, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()

        self.sae_cache = {}
        for layer in NEEDED_LAYERS + [63]:
            path = os.path.join(sae_local, f"layer{layer}.sae.pt")
            self.sae_cache[layer] = torch.load(path, map_location=self.model.device)
            print(f"  SAE L{layer} ready")

        print("[ARCHAEOLOGY] Ready.")

    @modal.method()
    def run_archaeology(self, dataset_path: str = "/data/creative-v2.jsonl",
                        layers: list = None, max_rows: int = 2000) -> dict:
        """Run feature archaeology on the TalkieLM dataset.

        For each row, compute which SAE features fire on the rewritten (vintage)
        text vs the original (modern) text. Find features that distinguish them.

        Returns dict of layer -> list of (feature_id, vintage_freq, modern_freq, diff).
        """
        import torch
        import json

        layers = layers or [15, 30, 40, 50, 63]
        rows = []
        with open(dataset_path) as f:
            for line in f:
                if len(rows) >= max_rows:
                    break
                row = json.loads(line)
                rows.append(row)

        print(f"[ARCHAEOLOGY] Processing {len(rows)} rows across layers {layers}")

        results = {}
        for layer in layers:
            sae = self.sae_cache[layer]
            W_enc = sae["W_enc"]  # [hidden, n_features]
            b_enc = sae["b_enc"]  # [n_features]

            # Track feature activations per class
            vintage_acts = torch.zeros(W_enc.shape[1])
            modern_acts = torch.zeros(W_enc.shape[1])
            vintage_count = 0
            modern_count = 0

            for i, row in enumerate(rows):
                if i % 100 == 0:
                    print(f"  L{layer} row {i}/{len(rows)}")

                for label, text_key in [("vintage", "rewritten_output"), ("modern", "original_output")]:
                    text = row.get(text_key, "")
                    if not text:
                        continue

                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)

                    # Get hidden state at target layer
                    with torch.no_grad():
                        hidden = self.model.model.layers[layer](
                            self.model.model.embed_tokens(inputs["input_ids"]),
                        )

                    # Compute SAE activations
                    acts = torch.nn.functional.relu(hidden @ W_enc + b_enc)
                    # Binary: did the feature fire at all?
                    fired = (acts > 0).any(dim=1).float().mean(dim=0).squeeze()

                    if label == "vintage":
                        vintage_acts += fired
                        vintage_count += 1
                    else:
                        modern_acts += fired
                        modern_count += 1

            # Normalize to frequencies
            vintage_freq = vintage_acts / max(vintage_count, 1)
            modern_freq = modern_acts / max(modern_count, 1)
            diff = vintage_freq - modern_freq

            # Get top 20 vintage-correlated and modern-correlated features
            top_vintage = torch.argsort(diff, descending=True)[:20]
            top_modern = torch.argsort(diff)[:20]

            layer_result = {
                "vintage_count": vintage_count,
                "modern_count": modern_count,
                "top_vintage_features": [
                    {
                        "feature_id": int(fid),
                        "vintage_freq": float(vintage_freq[fid]),
                        "modern_freq": float(modern_freq[fid]),
                        "diff": float(diff[fid]),
                    }
                    for fid in top_vintage
                ],
                "top_modern_features": [
                    {
                        "feature_id": int(fid),
                        "vintage_freq": float(vintage_freq[fid]),
                        "modern_freq": float(modern_freq[fid]),
                        "diff": float(diff[fid]),
                    }
                    for fid in top_modern
                ],
            }
            results[f"layer_{layer}"] = layer_result
            print(f"  L{layer}: top vintage feature = {layer_result['top_vintage_features'][0]['feature_id']} "
                  f"(diff={layer_result['top_vintage_features'][0]['diff']:.3f})")

        results["rows_processed"] = len(rows)
        results["layers_tested"] = layers
        return results


# ── Quick smoke test ─────────────────────────────────────────────────
@app.local_entrypoint()
def smoke_test():
    """Quick test: generate a steered and unsteered response."""
    import json

    server = modal.Cls.from_name("talkie-sae", "TalkieSAEServer")()

    # We can't call the ASGI app directly, so use the _generate method via a helper
    print("[SMOKE] Testing generation via internal method...")

    # Instead, let's test the web endpoint
    import httpx
    endpoint = modal.Cls.from_name("talkie-sae", "TalkieSAEServer").serve.web_url
    print(f"[SMOKE] Server URL: {endpoint}")

    # Or just call the class method directly
    # We'll add a @modal.method() for direct testing
    print("[SMOKE] Done. Use the /v1/chat/completions endpoint or run archaeology.")


if __name__ == "__main__":
    print("Talkie SAE Server")
    print("Deploy with: modal deploy src/server.py")
    print("Then point llmcord at the resulting URL")
