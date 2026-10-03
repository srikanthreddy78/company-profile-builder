"""The agent's tools: a stable facade over the cohesive modules the tool logic lives in.

- `agent.context`   ToolContext + FatalProfileError
- `agent.ingest`    page validation/caching/indexing and the scrape-loop helpers
- `agent.evidence`  excerpt/answer verification and grounding
- `agent.drafting`  contract validation, repair loop, draft merge and commit
- `agent.interview` question identity and answer classification
- `agent.finalize`  omission of disputed/ungrounded values and the export

The tool functions themselves (whose docstrings the model reads) are grouped in
`web_tools`, `interview_tools` and `draft_tools`; `make_tools` assembles them in the
order the agent is given.
"""

from __future__ import annotations

from typing import Any

from profile_builder.agent.context import FatalProfileError, ToolContext
from profile_builder.agent.evidence import ungrounded_paths
from profile_builder.agent.finalize import finalize, no_usable_pages_message, omit_many
from profile_builder.agent.ingest import process_page
from profile_builder.agent.interview import is_multi_question
from profile_builder.agent.tools.draft_tools import make_draft_tools
from profile_builder.agent.tools.interview_tools import make_interview_tools
from profile_builder.agent.tools.web_tools import make_web_tools
from profile_builder.config import SCRAPE_INTER_REQUEST_DELAY_S
from profile_builder.web.scraper import is_transient_code

__all__ = [
    "SCRAPE_INTER_REQUEST_DELAY_S",
    "FatalProfileError",
    "ToolContext",
    "finalize",
    "is_multi_question",
    "is_transient_code",
    "make_tools",
    "no_usable_pages_message",
    "omit_many",
    "process_page",
    "ungrounded_paths",
]


def make_tools(ctx: ToolContext) -> list[Any]:
    """discover_pages, scrape_pages, search_pages, read_page, ask_user, save_profile_draft,
    apply_profile_updates, note_conflict, finalize_profile."""
    return [*make_web_tools(ctx), *make_interview_tools(ctx), *make_draft_tools(ctx)]
