"""Assemble model, embeddings, checkpointer and the deep agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents._models import get_model_provider
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.profiles.harness import harness_profiles as _hp
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from profile_builder.agent.middleware import build_middleware
from profile_builder.agent.prompts import render_system_prompt
from profile_builder.agent.tools import ToolContext, make_tools
from profile_builder.config import CHECKPOINT_DB, MODEL_REQUEST_TIMEOUT_S, Settings
from profile_builder.retrieval.index import Embeddings

# Read-only virtual filesystem: the agent can inspect evicted large tool results but has
# no write/execute surface. StateBackend keeps everything in checkpointed state (no disk).
FILESYSTEM_TOOLS = ["ls", "read_file"]


def build_model(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.has_openai():
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env or the environment)")
    # max_retries=0: the ModelRetryMiddleware owns retries/backoff so attempts stay bounded.
    return ChatOpenAI(
        model=settings.model_id,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        timeout=MODEL_REQUEST_TIMEOUT_S,
        max_retries=0,
    )


def build_embeddings(settings: Settings) -> Embeddings | None:
    if not settings.use_embeddings or not settings.has_openai():
        return None
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        timeout=MODEL_REQUEST_TIMEOUT_S,
        max_retries=settings.max_retries,
    )


def build_checkpointer(run_dir: Path) -> SqliteSaver:
    path = run_dir / CHECKPOINT_DB
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover
        pass
    return saver


def disable_default_subagent(model: BaseChatModel) -> None:
    """Deep Agents auto-adds a `task` tool + general-purpose subagent that would run our tools
    *without* our middleware (limits, budget, serial calls, untrusted-content framing). The
    harness profile is resolved per model provider, so register an override for this model's
    provider that disables it (merging with any existing provider profile)."""
    import dataclasses

    provider = get_model_provider(model) or "unknown"
    existing = _hp._get_harness_profile(provider) or HarnessProfile()
    register_harness_profile(
        provider,
        dataclasses.replace(
            existing, general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )


def exposed_tool_names(agent) -> set[str]:
    node = agent.nodes.get("tools")
    bound = getattr(node, "bound", None) or node
    return set(getattr(bound, "tools_by_name", {}).keys())


def build_agent(ctx: ToolContext, model: BaseChatModel, checkpointer: SqliteSaver):
    settings = ctx.settings
    disable_default_subagent(model)
    backend = StateBackend()
    middleware = [
        FilesystemMiddleware(
            backend=backend, tools=FILESYSTEM_TOOLS
        ),  # replaces the default by name
        *build_middleware(settings, ctx.store),
    ]
    agent = create_deep_agent(
        model=model,
        tools=make_tools(ctx),
        system_prompt=render_system_prompt(
            max_pages=settings.max_pages,
            max_questions=settings.max_questions,
            product_focus=ctx.product_focus,
        ),
        middleware=middleware,
        backend=backend,
        checkpointer=checkpointer,
        name="company-profile-builder",
    )
    exposed = exposed_tool_names(agent)
    if "task" in exposed or "execute" in exposed:
        raise RuntimeError(f"unexpected agent tools exposed: {sorted(exposed)}")
    return agent
