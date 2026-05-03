"""
Talkie SAE llmcord Fork — Discord Bot with SAE-Steered Qwen3.5-27B
===================================================================
A patched llmcord that connects to the Talkie SAE Modal server instead
of a local Talkie MLX server.

Key differences from upstream:
  - Cold start awareness: first request may take ~90s, subsequent ~3-5s
  - Native vision support (no VLM proxy needed — Qwen3.5 sees directly)
  - /model command includes talkie-sae option
  - Timeout handling: 120s timeout for Modal cold starts

Based on: github.com/jakobdylanc/llmcord (lightly patched)
Talkie original: github.com/gwyntel/talkie-mlx-server
"""

import asyncio
from base64 import b64encode
from collections import deque
import json
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
import time
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# ── Model tags that support vision ──────────────────────────────────
# Qwen3.5-27B is a native VLM, so talkie-sae supports images
VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4",
                     "llama", "llava", "mistral", "o3", "o4", "qwen", "vision", "vl")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

# Modal cold start can be ~90s; give it time
MODAL_TIMEOUT = 120

MAX_MESSAGE_NODES = 500

# ── Era-format <-> Discord ping translator ──────────────────────────
_CORRESPONDENT_PATTERN = re.compile(r"Correspondent No\.\s*(\d+)(:)")

def era_to_discord(text: str) -> str:
    """Convert era-appropriate 'Correspondent No. N:' back to Discord <@N> pings."""
    return _CORRESPONDENT_PATTERN.sub(r"<@\1>:", text)


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

# Per-channel/thread mutable context tracking
t_mutable_context = {}  # channel_id -> {opener_id, opened_at, ttl_seconds}
MUTABLE_CONTEXT_TTL = 900  # 15 minutes

# ── Token output tracking + uptime (for dynamic bot status) ─────────
_bot_start_time = time.time()
_output_log: deque = deque()
_STATUS_UPDATE_INTERVAL = 60
_TOKEN_WINDOW_SECONDS = 6 * 3600  # 6h rolling window

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "Talking like it's 1929")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient(timeout=MODAL_TIMEOUT)


# ── Dynamic status updater ──────────────────────────────────────────
def _format_uptime(seconds: float) -> str:
    s = int(seconds)
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{m}m"
    h, m = divmod(m, 60)
    if h < 24: return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def _format_tokens(count: int) -> str:
    if count < 1000: return str(count)
    elif count < 100_000: return f"{count / 1000:.1f}k"
    else: return f"{count / 1000:.0f}k"


def _build_status_text() -> str:
    base = config.get("status_message") or "Talking like it's 1929"
    now = time.time()
    while _output_log and (now - _output_log[0][0]) > _TOKEN_WINDOW_SECONDS:
        _output_log.popleft()
    total_tokens = sum(t[1] for t in _output_log)
    uptime = _format_uptime(now - _bot_start_time)
    token_str = _format_tokens(total_tokens)
    status = f"{base} · {token_str} tok/6h · up {uptime}"
    return status[:128]


async def _status_updater():
    await discord_bot.wait_until_ready()
    while not discord_bot.is_closed():
        try:
            text = _build_status_text()
            activity = discord.CustomActivity(name=text)
            await discord_bot.change_presence(activity=activity)
        except Exception:
            logging.exception("[status_updater] failed")
        await asyncio.sleep(_STATUS_UPDATE_INTERVAL)


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ── Slash Commands ──────────────────────────────────────────────────
@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model
    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."
    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config
    if curr_str == "":
        config = await asyncio.to_thread(get_config)
    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]
    return choices[:25]


@discord_bot.tree.command(name="context", description="View context, clear memory cache, or show bot status")
async def context_command(interaction: discord.Interaction, action: str) -> None:
    channel = interaction.channel

    if action == "status":
        max_messages = config.get("max_messages", 25)
        cached_nodes = len(msg_nodes)
        uptime = _format_uptime(time.time() - _bot_start_time)
        now_ts = time.time()
        recent_tokens = sum(t[1] for t in _output_log if now_ts - t[0] < _TOKEN_WINDOW_SECONDS)
        tok_str = _format_tokens(recent_tokens)

        # Check server health
        provider = curr_model.split("/")[0]
        provider_config = config["providers"].get(provider, {})
        base_url = provider_config.get("base_url", "").rstrip("/")
        online = False
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                resp = await c.get(f"{base_url}/v1/health")
                online = resp.status_code == 200
        except Exception:
            pass

        lines = [
            f"**📊 Bot Status**\n",
            f"```\n"
            f"Model:            {curr_model}\n"
            f"Max messages:      {max_messages}\n"
            f"Msg cache:         {cached_nodes} / {MAX_MESSAGE_NODES} nodes\n"
            f"Server:           {'✅ online' if online else '❌ offline'}\n"
            f"Timeout:          {MODAL_TIMEOUT}s (Modal cold start aware)\n"
            f"```\n",
            f"⏱️ Uptime: **{uptime}** — Output: **{tok_str} tokens** in last 6h",
        ]
        content = "\n".join(lines)
        await interaction.response.send_message(content, ephemeral=True)


@context_command.autocomplete("action")
async def context_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    actions = [
        Choice(name="📊 Status", value="status"),
        Choice(name="🗑️ Clear memory cache", value="clear"),
        Choice(name="👀 Show context", value="show"),
    ]
    return [a for a in actions if curr_str.lower() in a.name.lower()]


# ── Message handler ─────────────────────────────────────────────────
@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    # Don't respond to self
    if new_msg.author == discord_bot.user:
        return

    # Check if bot is mentioned or if it's a DM
    is_dm = new_msg.channel.type == discord.ChannelType.private
    is_mentioned = discord_bot.user.mention in new_msg.content

    if not is_dm and not is_mentioned:
        return

    # Check permissions
    if not is_dm:
        if new_msg.channel.id in config.get("permissions", {}).get("channels", {}).get("blocked_ids", []):
            return
        allowed_channels = config.get("permissions", {}).get("channels", {}).get("allowed_ids", [])
        if allowed_channels and new_msg.channel.id not in allowed_channels:
            return

    # React with typing indicator
    async with new_msg.channel.typing():
        await _handle_message(new_msg)


async def _handle_message(new_msg: discord.Message) -> None:
    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]
    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-needed")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=MODAL_TIMEOUT)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = []

                # ── Image handling ──────────────────────────────
                # Qwen3.5-27B has native vision — send images directly
                # No VLM proxy needed (unlike text-only Talkie 13B)
                image_attachments = [
                    (att, resp)
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                if image_attachments and accept_images:
                    # Pass images as base64 — the server handles VL natively
                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                        for att, resp in image_attachments
                    ]
                elif image_attachments and not accept_images:
                    # Model doesn't support vision — describe as text
                    photo_lines = [
                        f"[Correspondent sends a photograph, but the wireless is too faint to make it out.]"
                        for _ in image_attachments
                    ]
                    if photo_lines:
                        curr_node.text = (curr_node.text + "\n" + "\n".join(photo_lines)) if curr_node.text else "\n".join(photo_lines)

                if curr_node.role == "user" and (curr_node.text or curr_node.images):
                    curr_node.text = f"Correspondent No. {curr_msg.author.id}: {curr_node.text}"

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

            if curr_node.images[:max_images]:
                content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                messages.append(dict(role=curr_node.role, content=content))

            user_warnings.add("text truncated") if len(curr_node.text or "") > max_text else None
            user_warnings.add("images truncated") if len(curr_node.images) > max_images else None
            user_warnings.add("attachments unsupported") if curr_node.has_bad_attachments else None

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        messages.append(dict(role="system", content=system_prompt))

    # Generate response
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=False, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    logging.info(f"[API_CALL] model={model} messages_count={len(messages[::-1])}")

    use_plain_responses = config.get("use_plain_responses", False)
    max_message_length = 4000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    if not use_plain_responses:
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)
        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            response = await openai_client.chat.completions.create(**openai_kwargs)

        raw_text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        logging.info(f"[GENERATE] finish={finish_reason} raw_len={len(raw_text)}")

        # Record token output
        if hasattr(response, 'usage') and response.usage and response.usage.completion_tokens:
            est_tokens = response.usage.completion_tokens
        else:
            est_tokens = max(1, len(raw_text) // 4)
        _output_log.append((time.time(), est_tokens))

        translated = era_to_discord(raw_text)

        # Split into Discord-safe segments
        while translated:
            segment = translated[:max_message_length]
            response_contents.append(segment)
            translated = translated[max_message_length:]

        # Send segments
        for i, content in enumerate(response_contents):
            is_last = (i == len(response_contents) - 1)
            if use_plain_responses:
                await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))
            else:
                embed.description = content
                embed.color = EMBED_COLOR_COMPLETE if is_last else EMBED_COLOR_INCOMPLETE
                await reply_helper(embed=embed, silent=True if i > 0 else False)

    except Exception:
        logging.exception("Error while generating response")
        try:
            await new_msg.reply("⚠️ I say, the telegraph lines are rather tangled. The wireless operator may be warming up the apparatus — pray try once more in a moment.", silent=True)
        except Exception:
            pass
    else:
        for response_msg in response_msgs:
            msg_nodes[response_msg.id].text = "".join(response_contents)
            msg_nodes[response_msg.id].lock.release()

    # Clean old nodes
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
