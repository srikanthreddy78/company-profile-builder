"""Assemble model, embeddings, checkpointer and the deep agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
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


def build_agent(ctx: ToolContext, model: BaseChatModel, checkpointer: SqliteSaver):
    settings = ctx.settings
    backend = StateBackend()
    middleware = [
        FilesystemMiddleware(backend=backend, tools=FILESYSTEM_TOOLS),  # replaces the default by name
        *build_middleware(settings, ctx.store),
    ]
    return create_deep_agent(
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
