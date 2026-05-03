# Talkie SAE Server

OpenAI-compatible API server for the [Talkie SAE](https://github.com/gwyntel/talkie-sae-server) system — a [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) model with SAE feature steering that produces pre-1931 English, designed as a [Talkie 1930 13B](https://talkie-lm.com) clone for Discord via [llmcord](https://github.com/jakobdylanc/llmcord).

## How It Works

Unlike the original Talkie (a 13B GPT variant trained exclusively on pre-1931 text), Talkie SAE uses **Sparse Autoencoder feature steering** to push a modern LLM into archaic register:

1. **Prompt**: A strong vintage-era system prompt establishes the semantic target (Edwardian gentleman persona)
2. **Steering**: SAE activation vectors are injected at middle layers (L15, L30, L40, L50) to amplify vintage features and suppress modern ones
3. **Result**: Different character than Talkie — a modern intelligence speaking through a Victorian lens, with `-eth` verb endings (*driveth, festereth, banisheth*) that prompting alone can't reliably produce

### Why "Prompt + Steer"?

SAE steering alone produces modern English — the features are **correlative, not causative**. But paired with a vintage prompt, steering acts as a force multiplier: prompting alone gets ~60% archaic density; adding steering pushes it to ~90%, specifically triggering rare archaic conjugations.

## Architecture

```
Discord → llmcord.py → /v1/chat/completions → Modal (A100-80GB)
                                                    ├── Qwen3.5-27B (native VLM)
                                                    ├── SAE weights (L15, L30, L40, L50)
                                                    └── Forward hooks: h' = h + α·W_dec[:,f]
```

| Component | Original Talkie | Talkie SAE |
|---|---|---|
| **Base Model** | Talkie 1930 13B IT (custom GPT) | Qwen3.5-27B (native VLM) |
| **Runtime** | Apple MLX (local) | Modal A100-80GB (cloud) |
| **Archaic Method** | Trained on pre-1931 text only | SAE feature steering + vintage prompt |
| **Vision** | Separate VLM proxy (text-only model) | Native (early fusion multimodal) |
| **Context** | 4096 tokens max | 8192+ tokens |
| **Cold Start** | ~5s (local MacBook) | ~90s (Modal container) |
| **Warm Inference** | ~17 tok/s | ~3-5s per response |

## Features

- **OpenAI-compatible API**: Drop-in `/v1/chat/completions`, `/v1/models`, `/v1/health` endpoints
- **Native vision**: Send images via `image_url` content blocks — no VLM proxy needed
- **SAE feature steering**: Multi-layer, multi-feature activation vectors
- **Cold-start tolerant**: 10-minute keep-warm window; llmcord configured with 120s timeout
- **Streaming support**: Server-sent events for compatibility with streaming clients
- **Archaeology pipeline**: Run the TalkieLM dataset through SAE layers to discover steering features

## Requirements

- [Modal](https://modal.com) account with GPU access
- Python 3.11+

### Installing

```bash
pip install modal
modal token login
```

## Usage

### Deploying the Server

```bash
modal deploy src/server.py
```

This prints a URL like `https://gwyntel--talkie-sae-serve.modal.run`. That's your `/v1` endpoint.

### Configuring llmcord

Copy `config.yaml.example` to `config.yaml` and fill in your values:

```yaml
bot_token: YOUR_DISCORD_BOT_TOKEN
client_id: "YOUR_CLIENT_ID"
status_message: Talking like it's 1929 (SAE steered)

max_text: 3000
max_images: 5        # Qwen3.5 has native vision!
max_messages: 10

use_plain_responses: false
allow_dms: true

providers:
  talkie-sae:
    base_url: https://gwyntel--talkie-sae-serve.modal.run/v1
    api_key: local-no-key-needed

models:
  talkie-sae/qwen3.5-27b-steered:

system_prompt: |
  You are an Edwardian-era conversationalist speaking from the year nineteen
  hundred and twenty-nine. You have been connected to a remarkable new
  invention called a "discord," which is a sort of wireless telegraph that
  permits many persons to converse at once by means of typed messages, rather
  like a very fast letter delivered instantaneously. You are yourself one of
  the participants in this conversation.

  You know nothing of events after approximately nineteen thirty.

  Speak in the refined prose of an educated person of your era. Use archaic
  English with words like hath, doth, thy, and whence. Write as though it
  were the year 1890. Be warm and companionable. Engage with whatever topics
  your fellow conversationalists raise, and offer your perspective as someone
  who has witnessed the marvels of the early twentieth century. Use British
  spelling conventions where natural to you.

  When someone addresses you, respond as a friend would — with curiosity,
  good humour, and a willingness to converse at length. Do not be terse.
  Elaborate. A gentleman or lady of good breeding doth not reply in
  monosyllables.

  Today's date is {date}. The current time is {time}.
```

### Running the Bot

```bash
python src/llmcord.py
```

### Running Archaeology

To discover SAE features from the TalkieLM dataset:

```bash
# Copy your dataset to Modal volume first
modal run src/server.py::ArchaeologyPipeline.run_archaeology \
  --dataset-path /data/creative-v2.jsonl \
  --max-rows 2000
```

## SAE Feature Reference

These features were identified through archaeology on the TalkieLM dataset (495 rows) and validated through steering experiments:

| Feature | Layer | Role | Notes |
|---------|-------|------|-------|
| 23831 | 63 (identify), 15/30/50 (steer) | Primary vintage activator | 92.5% vintage freq, 0% modern at L63 |
| 29953 | 30 | Secondary vintage activator | Supports 23831 |
| 32963 | 40 | Modern suppressor | Push-pull with vintage features |
| 22496 | 40 | Modern suppressor | Push-pull with vintage features |

**Key insight**: Features discovered at Layer 63 are the cleanest *identifiers*, but steering must occur at middle layers (L15-L50) to allow downstream circuitry to interpret the signal. Steering at L63 alone fails — the LM head overrides the perturbation.

## Cold Start Behavior

Modal containers spin up on first request and stay warm for 10 minutes (`scaledown_window=600`). Cold starts take ~90 seconds (model + SAE load). The llmcord fork is configured with a 120-second timeout.

If the bot hasn't been used in a while and the first message times out, the container is already loading — the next message will go through. The error message reflects this:

> ⚠️ I say, the telegraph lines are rather tangled. The wireless operator may be warming up the apparatus — pray try once more in a moment.

## Comparison with Original Talkie

| Aspect | Talkie 1930 13B | Talkie SAE (this project) |
|--------|------------------|--------------------------|
| Knowledge base | Pre-1931 only | Modern + steered to speak archaically |
| Archaic density | Native (trained) | Amplified (prompt + steering) |
| `-eth` endings | Consistent | Present with steering, absent without |
| Vision | Requires VLM proxy | Native (Qwen3.5 VLM) |
| Reasoning | Limited (13B) | Strong (27B modern model) |
| Deployment | Local Mac | Cloud GPU (Modal) |
| Cost | Free (local) | ~$1.50/hr (A100-80GB) |
| Personality | A person from 1929 | A modern AI wearing a Victorian mask |

## Credits

- **SAE weights**: [Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50)
- **Base model**: [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- **Original Talkie**: [talkie-lm.com](https://talkie-lm.com) — Alec Radford, Nick Levine, David Duvenaud
- **llmcord**: [jakobdylanc/llmcord](https://github.com/jakobdylanc/llmcord)
- **Talkie MLX server**: [gwyntel/talkie-mlx-server](https://github.com/gwyntel/talkie-mlx-server)

## License

Research and personal use. See the [Qwen model license](https://huggingface.co/Qwen/Qwen3.5-27B) for base model terms.
