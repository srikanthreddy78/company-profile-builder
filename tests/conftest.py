"""Shared test harness: a scripted chat model (predetermined tool calls), a synthetic
fixture website, deterministic embeddings, and a Runner factory wired for offline use."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from rich.console import Console

from profile_builder.config import Settings
from profile_builder.retrieval.index import HashEmbeddings
from profile_builder.web.scraper import FixtureScraper, LinkCandidate, ScrapedPage
from profile_builder.workflow.runner import Runner

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """An AIMessage containing exactly one tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id
                or f"call_{name}_{abs(hash(json.dumps(args, sort_keys=True, default=str))) % 10_000_000}",
            }
        ],
    )


def last_tool_result(messages: list[BaseMessage]) -> dict[str, Any]:
    """Parse the JSON payload of the most recent ToolMessage (stripping untrusted framing)."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            text = m.content if isinstance(m.content, str) else json.dumps(m.content)
            lines = [ln for ln in text.splitlines() if not ln.startswith("<<<")]
            return json.loads("\n".join(lines))
    raise AssertionError("no ToolMessage found")


class ScriptedChatModel(BaseChatModel):
    """Returns predetermined AIMessages. Steps may be callables that inspect the
    conversation (e.g. to read a question id from the previous tool result)."""

    steps: list[Any]
    cursor: int = 0
    calls: int = 0
    fail_first_n: int = 0  # raise an exception on the first N calls (retry tests)
    failure: Exception | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any
    ) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            raise self.failure or RuntimeError("scripted model failure")
        if self.cursor >= len(self.steps):
            raise AssertionError(f"scripted model exhausted after {len(self.steps)} steps")
        step = self.steps[self.cursor]
        self.cursor += 1
        msg = step(messages) if callable(step) else step
        msg = msg.model_copy(
            update={
                "usage_metadata": {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
            }
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


# --------------------------------------------------------------------------------------
# Synthetic fixture website: acme-example.com
# --------------------------------------------------------------------------------------

SITE = "https://acme-example.com"

HOME_MD = """# Acme Vault — Confidential Data Platform

Acme Vault helps startups and enterprises protect sensitive data in use with confidential computing.
Our platform encrypts data while it is being processed, so even cloud providers cannot read it.

## Why Acme

Trusted by security teams at fast-moving startups and global enterprises alike.

## Products

- Acme Vault: confidential data platform
- Acme Keys: key management service
"""

PRODUCT_MD = """# Acme Vault

Acme Vault is a confidential data platform that keeps data encrypted in use, at rest and in transit.

## How it works

Workloads run inside hardware-backed secure enclaves. Keys never leave the enclave boundary, and every access is logged for audit.

## Capabilities

### Runtime encryption
Runtime encryption protects data while it is processed in memory. It uses Intel SGX and AMD SEV enclaves. Customers can run sensitive analytics without exposing plaintext to the cloud provider.

### Policy-based key release
Keys are released only to attested workloads that match a policy. This stops stolen credentials from decrypting data.

## Differentiators

Unlike traditional encryption tools, Acme Vault protects data during computation, not only at rest.
"""

CUSTOMERS_MD = """# Customers

Acme Vault is used by Fortune 500 banks and healthcare networks to meet strict data residency rules.

## Case study: Northwind Bank

Northwind Bank reduced its compliance audit time by 40% after moving fraud analytics into Acme Vault enclaves.

## Case study: Contoso Health

Contoso Health shares patient data with research partners without exposing raw records.
"""

ABOUT_MD = """# About Acme

Founded in 2019 by former cloud security engineers, Acme builds privacy-preserving infrastructure.

Our team has published research on enclave attestation and contributes to the Confidential Computing Consortium.
"""

PRICING_MD = """# Pricing

Acme Vault pricing is based on the number of protected workloads. Contact sales for an enterprise quote.
"""

PAGES: dict[str, tuple[str, str]] = {
    f"{SITE}/": ("Acme Vault — Confidential Data Platform", HOME_MD),
    f"{SITE}/product": ("Acme Vault product", PRODUCT_MD),
    f"{SITE}/customers": ("Customers", CUSTOMERS_MD),
    f"{SITE}/about": ("About Acme", ABOUT_MD),
    f"{SITE}/pricing": ("Pricing", PRICING_MD),
}


@pytest.fixture(scope="session")
def acme_fixtures(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("acme")
    for url, (title, md) in PAGES.items():
        FixtureScraper.write_page(
            root,
            ScrapedPage(
                url=url,
                final_url=url,
                title=title,
                description="",
                markdown=md,
                http_status=200,
                links=[u for u in PAGES],
            ),
        )
    FixtureScraper.write_map(
        root,
        [LinkCandidate(url=u, title=t) for u, (t, _) in PAGES.items()]
        + [
            LinkCandidate(url=f"{SITE}/blog/post-1", title="Blog"),
            LinkCandidate(url=f"{SITE}/careers", title="Careers"),
        ],
    )
    return root


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    import profile_builder.agent.middleware as mw

    monkeypatch.setattr(mw, "RETRY_INITIAL_DELAY_S", 0.0)
    monkeypatch.setattr(mw, "RETRY_MAX_DELAY_S", 0.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=None,
        firecrawl_api_key=None,
        runs_dir=tmp_path / "runs",
        max_pages=4,
        max_questions=3,
    )


def make_runner(
    settings: Settings,
    steps: list[Step],
    *,
    scraper: FixtureScraper,
    answers: list[str] | None = None,
    model: ScriptedChatModel | None = None,
    non_interactive: bool = False,
    **kwargs: Any,
) -> tuple[Runner, ScriptedChatModel, list[str]]:
    model = model or ScriptedChatModel(steps=steps)
    queue = list(answers or [])
    asked: list[str] = []

    def input_fn(prompt: str) -> str:
        if not queue:
            raise EOFError
        ans = queue.pop(0)
        asked.append(ans)
        return ans

    runner = Runner(
        settings,
        Console(record=True, width=120, force_terminal=False, no_color=True),
        model_factory=lambda s: model,
        scraper_factory=lambda s: scraper,
        embeddings_factory=lambda s: HashEmbeddings(),
        input_fn=input_fn,
        non_interactive=non_interactive,
        check_dns=False,
        **kwargs,
    )
    return runner, model, asked


# --------------------------------------------------------------------------------------
# A full happy-path script (reused by several tests)
# --------------------------------------------------------------------------------------

DRAFT_PROFILE: dict[str, Any] = {
    "artifact": "company_brain",
    "version": 1,
    "company": {"name": "Acme", "website_url": f"{SITE}/"},
    "product": {
        "name": "Acme Vault",
        "description": "A confidential data platform that keeps data encrypted in use, at rest and in transit.",
        "positioning": "Protects data during computation, not only at rest.",
        "features_and_capabilities": [
            {
                "name": "Runtime encryption",
                "description": "Protects data while it is processed in memory.",
                "how_it_works": "Uses Intel SGX and AMD SEV enclaves.",
                "customer_benefit": "Run sensitive analytics without exposing plaintext to the cloud provider.",
            },
            {
                "name": "Policy-based key release",
                "description": "Keys are released only to attested workloads that match a policy.",
                "how_it_works": "",
                "customer_benefit": "Stops stolen credentials from decrypting data.",
            },
        ],
        "differentiators": ["Protects data during computation, not only at rest"],
    },
    "customer": {
        "target_customer": "Startups and enterprises protecting sensitive data",
        "buyers": [],
        "users": ["Security teams"],
        "problems": ["Cloud providers can read data while it is processed"],
        "use_cases": ["Fraud analytics in enclaves", "Sharing patient data with research partners"],
        "desired_outcomes": ["Reduced compliance audit time"],
        "existing_alternatives": ["Traditional encryption tools"],
    },
    "content_evidence": {
        "customer_stories": [
            "Northwind Bank reduced compliance audit time by 40%",
            "Contoso Health shares patient data without exposing raw records",
        ],
        "company_expertise": [
            "Published research on enclave attestation; Confidential Computing Consortium contributor"
        ],
        "product_evidence": [],
        "proprietary_insights_or_examples": [],
    },
    "brand": {
        "voice_and_tone": [],
        "writing_style": [],
        "preferred_terms": ["confidential computing"],
        "terms_or_claims_to_avoid": [],
    },
}

DRAFT_EVIDENCE: list[dict[str, str]] = [
    {
        "field_path": "product.name",
        "source_url": f"{SITE}/product",
        "excerpt": "Acme Vault is a confidential data platform",
    },
    {
        "field_path": "product.description",
        "source_url": f"{SITE}/product",
        "excerpt": "Acme Vault is a confidential data platform that keeps data encrypted in use, at rest and in transit.",
    },
    {
        "field_path": "product.positioning",
        "source_url": f"{SITE}/product",
        "excerpt": "Acme Vault protects data during computation, not only at rest.",
    },
    {
        "field_path": "product.features_and_capabilities[0]",
        "source_url": f"{SITE}/product",
        "excerpt": "Runtime encryption protects data while it is processed in memory. It uses Intel SGX and AMD SEV enclaves.",
    },
    {
        "field_path": "product.features_and_capabilities[1]",
        "source_url": f"{SITE}/product",
        "excerpt": "Keys are released only to attested workloads that match a policy. This stops stolen credentials from decrypting data.",
    },
    {
        "field_path": "product.differentiators",
        "source_url": f"{SITE}/product",
        "excerpt": "Unlike traditional encryption tools, Acme Vault protects data during computation, not only at rest.",
    },
    {
        "field_path": "customer.target_customer",
        "source_url": f"{SITE}/",
        "excerpt": "Acme Vault helps startups and enterprises protect sensitive data in use",
    },
    {
        "field_path": "customer.users",
        "source_url": f"{SITE}/",
        "excerpt": "Trusted by security teams at fast-moving startups and global enterprises alike.",
    },
    {
        "field_path": "customer.problems",
        "source_url": f"{SITE}/",
        "excerpt": "so even cloud providers cannot read it",
    },
    {
        "field_path": "customer.use_cases",
        "source_url": f"{SITE}/customers",
        "excerpt": "moving fraud analytics into Acme Vault enclaves",
    },
    {
        "field_path": "customer.desired_outcomes",
        "source_url": f"{SITE}/customers",
        "excerpt": "reduced its compliance audit time by 40%",
    },
    {
        "field_path": "customer.existing_alternatives",
        "source_url": f"{SITE}/product",
        "excerpt": "Unlike traditional encryption tools",
    },
    {
        "field_path": "content_evidence.customer_stories[0]",
        "source_url": f"{SITE}/customers",
        "excerpt": "Northwind Bank reduced its compliance audit time by 40%",
    },
    {
        "field_path": "content_evidence.customer_stories[1]",
        "source_url": f"{SITE}/customers",
        "excerpt": "Contoso Health shares patient data with research partners without exposing raw records.",
    },
    {
        "field_path": "content_evidence.company_expertise",
        "source_url": f"{SITE}/about",
        "excerpt": "published research on enclave attestation and contributes to the Confidential Computing Consortium",
    },
    {
        "field_path": "brand.preferred_terms",
        "source_url": f"{SITE}/",
        "excerpt": "confidential computing",
    },
    # deliberately unverifiable → must be rejected
    {
        "field_path": "company.name",
        "source_url": f"{SITE}/about",
        "excerpt": "Acme is the market leader in everything",
    },
]

CONFLICT_QUESTION = {
    "question": "The homepage says Acme serves startups and enterprises, but the customers page only shows Fortune 500 banks and healthcare networks. Which segment should this profile prioritize?",
    "why_unclear": "Two pages describe different target customers.",
    "field_paths": ["customer.target_customer"],
    "kind": "conflict",
}


def happy_path_steps(*, ask: bool = True, finalize: bool = True) -> list[Step]:
    steps: list[Step] = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages",
            {
                "urls": [
                    f"{SITE}/product",
                    f"{SITE}/customers",
                    f"{SITE}/about",
                    f"{SITE}/pricing",
                    f"{SITE}/blog/post-1",
                ]
            },
        ),
        tool_call("search_pages", {"query": "target customers banks healthcare"}),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
    ]
    if ask:
        steps.append(
            tool_call(
                "note_conflict",
                {
                    "field_path": "customer.target_customer",
                    "claims": [
                        {
                            "source_url": f"{SITE}/",
                            "excerpt": "helps startups and enterprises protect sensitive data",
                            "claim": "startups and enterprises",
                        },
                        {
                            "source_url": f"{SITE}/customers",
                            "excerpt": "used by Fortune 500 banks and healthcare networks",
                            "claim": "large regulated enterprises",
                        },
                    ],
                    "summary": "homepage vs customers page disagree on the target segment",
                },
            )
        )
        steps.append(tool_call("ask_user", CONFLICT_QUESTION))

        def apply_answer(messages: list[BaseMessage]) -> AIMessage:
            res = last_tool_result(messages)
            if res.get("status") != "answered":
                return (
                    tool_call("finalize_profile", {})
                    if finalize
                    else AIMessage(content="done without finalize")
                )
            return tool_call(
                "apply_profile_updates",
                {
                    "updates": [
                        {
                            "field_path": "customer.target_customer",
                            "value": res["answer"],
                            "evidence": {"kind": "interview", "question_id": res["qid"]},
                        }
                    ]
                },
            )

        steps.append(apply_answer)
    if finalize:
        steps.append(tool_call("finalize_profile", {}))
    steps.append(AIMessage(content="Profile exported."))
    return steps
