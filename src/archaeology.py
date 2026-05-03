"""
Talkie SAE Archaeology — Full 2k Row Run
==========================================
Run feature archaeology on the full creative-v2.jsonl dataset (1951 rows)
across layers 15, 30, 40, 50, and 63.

This uses the proven hook-capture approach from layer63_pipeline.py:
  1. Register a forward hook on the target layer
  2. Run a forward pass through the full model
  3. Capture the residual stream at that layer
  4. Compute SAE activations (ReLu(residual @ W_enc + b_enc))
  5. Track which features fire on vintage vs modern text

Output: Per-layer top vintage-correlated and modern-correlated features
with frequencies, average activations, and differentials.

Stats: ~1951 rows × 2 texts × 5 layers = ~19,510 forward passes
Time:  ~20-30 min on A100-80GB
"""

import modal
import json

app = modal.App("talkie-sae")

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
    )
)

MODEL_DIR = "/models"
model_vol = modal.Volume.from_name("qwen-scope-models", create_if_missing=True)
BASE_MODEL = "Qwen/Qwen3.5-27B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"
ALL_LAYERS = [15, 30, 40, 50, 63]


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={MODEL_DIR: model_vol},
    timeout=3600,  # 1 hour — 2k rows takes a while
    scaledown_window=30,
)
class Archaeology:
    @modal.enter()
    def setup(self):
        import os
        import subprocess
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_local = os.path.join(MODEL_DIR, "qwen3.5-27b")
        sae_local = os.path.join(MODEL_DIR, "qwen3.5-27b-sae")

        # Ensure base model exists
        if not os.path.exists(os.path.join(model_local, "config.json")):
            print("[SETUP] Downloading base model...")
            subprocess.run(["hf", "download", BASE_MODEL, "--local-dir", model_local], check=True)
            model_vol.commit()

        # Download all SAE layers
        for layer in ALL_LAYERS:
            path = os.path.join(sae_local, f"layer{layer}.sae.pt")
            if not os.path.exists(path):
                print(f"[SETUP] Downloading SAE L{layer}...")
                subprocess.run(
                    ["hf", "download", SAE_REPO, f"layer{layer}.sae.pt",
                     "--local-dir", sae_local], check=True,
                )
                model_vol.commit()

        print("[SETUP] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_local)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_local, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()

        self.sae_cache = {}
        for layer in ALL_LAYERS:
            path = os.path.join(sae_local, f"layer{layer}.sae.pt")
            self.sae_cache[layer] = torch.load(path, map_location=self.model.device)
            print(f"[SETUP] SAE L{layer}: W_enc={self.sae_cache[layer]['W_enc'].shape}")

        print("[SETUP] Ready.")

    def get_feature_activations(self, text: str, layer: int):
        """Get SAE feature activations for text at a specific layer.

        Returns list of {id, activation} for all active features (top-50).
        """
        import torch

        sae = self.sae_cache[layer]
        W_enc = sae["W_enc"]  # [hidden_dim, n_features]
        b_enc = sae["b_enc"]  # [n_features]

        # Capture residual stream via hook
        captured = {}
        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["residual"] = hidden.detach()

        hook_handle = self.model.model.layers[layer].register_forward_hook(_hook)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)

        with torch.no_grad():
            self.model(**inputs)
        hook_handle.remove()

        # Compute SAE activations at last token position
        residual = captured["residual"].to(W_enc.dtype)
        # residual: [1, seq_len, hidden_dim]
        # W_enc: [n_features, hidden_dim] — need transpose for matmul
        last_hidden = residual[0, -1, :]  # [hidden_dim]
        pre_acts = last_hidden @ W_enc.T + b_enc  # [n_features]
        acts = torch.nn.functional.relu(pre_acts)

        # Get all non-zero features
        active_mask = acts > 0
        active_ids = active_mask.nonzero(as_tuple=True)[0]
        result = [
            {"id": int(fid), "activation": float(acts[fid])}
            for fid in active_ids
        ]
        # Sort by activation strength descending
        result.sort(key=lambda x: x["activation"], reverse=True)
        return result

    @modal.method()
    def run_archaeology(self, pairs: list[dict]) -> dict:
        """Run full archaeology across all layers.

        Args:
            pairs: [{"modern": "...", "vintage": "..."}, ...]

        Returns:
            Per-layer feature differentials and top features.
        """
        from collections import defaultdict
        import time as _time

        total = len(pairs)
        results = {}

        print(f"\n{'='*60}")
        print(f"TALKIE SAE ARCHAEOLOGY: {total} pairs × {len(ALL_LAYERS)} layers")
        print(f"{'='*60}")

        for layer in ALL_LAYERS:
            t0 = _time.time()
            print(f"\n── Layer {layer} ──")

            modern_features = []
            vintage_features = []

            for i, pair in enumerate(pairs):
                m = self.get_feature_activations(pair["modern"], layer)
                v = self.get_feature_activations(pair["vintage"], layer)
                modern_features.append(m)
                vintage_features.append(v)

                if (i + 1) % 100 == 0:
                    elapsed = _time.time() - t0
                    eta = elapsed / (i + 1) * (total - i - 1)
                    print(f"  L{layer}: {i+1}/{total} pairs ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            # Differential analysis
            vintage_counts = defaultdict(float)
            modern_counts = defaultdict(float)
            vintage_values = defaultdict(list)
            modern_values = defaultdict(list)

            for m_feats, v_feats in zip(modern_features, vintage_features):
                for feat in m_feats:
                    fid = feat["id"]
                    modern_counts[fid] += 1
                    modern_values[fid].append(feat["activation"])
                for feat in v_feats:
                    fid = feat["id"]
                    vintage_counts[fid] += 1
                    vintage_values[fid].append(feat["activation"])

            differentials = []
            for fid in set(vintage_counts) | set(modern_counts):
                v_freq = vintage_counts.get(fid, 0) / total
                m_freq = modern_counts.get(fid, 0) / total
                v_vals = vintage_values.get(fid, [0])
                m_vals = modern_values.get(fid, [0])
                v_avg = sum(v_vals) / len(v_vals)
                m_avg = sum(m_vals) / len(m_vals)
                differentials.append({
                    "feature_id": fid,
                    "vintage_freq": round(v_freq, 4),
                    "modern_freq": round(m_freq, 4),
                    "vintage_avg_activation": round(v_avg, 4),
                    "modern_avg_activation": round(m_avg, 4),
                    "differential": round(v_freq - m_freq, 4),
                })

            differentials.sort(key=lambda x: x["differential"], reverse=True)

            top_vintage = differentials[:20]
            top_modern = differentials[-20:]

            elapsed = _time.time() - t0
            print(f"\n  L{layer} complete in {elapsed:.0f}s")
            print(f"  Top 3 VINTAGE features:")
            for f in top_vintage[:3]:
                print(f"    Feature {f['feature_id']}: diff={f['differential']:.3f} "
                      f"(vintage={f['vintage_freq']:.1%}, modern={f['modern_freq']:.1%})")
            print(f"  Top 3 MODERN features:")
            for f in top_modern[-3:]:
                print(f"    Feature {f['feature_id']}: diff={f['differential']:.3f} "
                      f"(vintage={f['vintage_freq']:.1%}, modern={f['modern_freq']:.1%})")

            results[f"layer_{layer}"] = {
                "total_pairs": total,
                "elapsed_seconds": round(elapsed, 1),
                "top_vintage": top_vintage,
                "top_modern": top_modern,
                "all_features_top50": differentials[:50],
            }

        # Summary across layers
        print(f"\n{'='*60}")
        print("CROSS-LAYER SUMMARY")
        print(f"{'='*60}")
        for layer in ALL_LAYERS:
            lr = results[f"layer_{layer}"]
            top_v = lr["top_vintage"][0]
            top_m = lr["top_modern"][-1]
            print(f"  L{layer}: top vintage={top_v['feature_id']} (diff={top_v['differential']:.3f}), "
                  f"top modern={top_m['feature_id']} (diff={top_m['differential']:.3f})")

        results["metadata"] = {
            "total_pairs": total,
            "layers_tested": ALL_LAYERS,
            "dataset": "creative-v2.jsonl",
        }
        return results


@app.local_entrypoint()
def main():
    import os

    # Load the full 2k dataset
    data_path = "/home/hermes/creative-v2.jsonl"
    pairs = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            pairs.append({
                "modern": row["original_output"],
                "vintage": row["rewritten_output"],
            })

    print(f"Loaded {len(pairs)} pairs from {data_path}")

    # Run archaeology — use the class directly since it's defined in this app
    result = Archaeology().run_archaeology.remote(pairs)

    # Save results
    out_dir = "/home/hermes/talkie-sae-server/cache"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "archaeology_2k_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")

    # Print cross-layer summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS: Top vintage feature per layer")
    print(f"{'='*60}")
    for layer in ALL_LAYERS:
        lr = result[f"layer_{layer}"]
        for f in lr["top_vintage"][:5]:
            marker = "★" if f["feature_id"] == 23831 else " "
            print(f"  L{layer} {marker} F{f['feature_id']:>6d}: diff={f['differential']:.3f} "
                  f"vintage={f['vintage_freq']:.1%} modern={f['modern_freq']:.1%}")
