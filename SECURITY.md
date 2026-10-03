# Security notes

This tool fetches arbitrary public web pages, feeds them to an LLM, and writes files. The
controls below are implemented in code and covered by tests (`tests/test_url_guard.py`,
`tests/test_logging_and_security.py`, `tests/test_happy_path.py`).

| Threat | Control | Where |
|---|---|---|
| **SSRF** via the start URL, discovered links, or redirects | `http(s)` only; no embedded credentials; default ports only; URL length cap; the hostname is resolved and **every** A/AAAA answer must be publicly routable (no loopback, RFC1918, link-local incl. cloud metadata `169.254.169.254`, CGNAT `100.64/10`, multicast, reserved, unspecified, IPv6 `::1`/`fc00::/7`/`fe80::/10`, IPv4-mapped, 6to4/Teredo). Numeric hosts in decimal/hex/octal forms are rejected. The **final URL after redirects** reported by Firecrawl is re-checked and must stay on the same registrable domain. `robots.txt` is fetched through the same guard with a timeout. | `web/url_guard.py`, `agent/tools.py::process_page`, `web/robots.py` |
| **Prompt injection** from scraped content | The agent has no action tools beyond the run's own SQLite store (read-only virtual filesystem, no shell). Web-derived tool output is wrapped in explicit "untrusted content" delimiters by middleware; the system prompt states page text is data. Instruction-like phrases are detected and logged as `INJECTION_SUSPECTED` without being acted on. | `agent/middleware.py::UntrustedContentMiddleware`, `security.py::detect_injection`, `agent/prompts.py` |
| **Fabricated facts, URLs or quotations** | Every website evidence row must name a page fetched in this run and quote a verbatim (whitespace/markdown-normalized) substring of its cached text; otherwise it is rejected and the model is told why. Field paths are validated against the contract allowlist and values are re-validated with Pydantic after every update. Unresolved conflicts are omitted from the export. Ungrounded fields are reported in `report.md` and `evidence.json`. | `agent/tools.py::_verify_website_evidence`, `schema.py::parse_field_path`, `agent/tools.py::finalize` |
| **Secret leakage** | Keys are `SecretStr` loaded from env/`.env` only (never CLI args). A redaction filter masks the loaded keys and `sk-…`/`fc-…`/`Bearer …` patterns in every log record, including exception text. `.env` is git-ignored; `doctor`/`status` show presence only. | `config.py`, `logging_setup.py::RedactSecretsFilter` |
| **Path traversal** via `--run-id` / `--runs-dir` | Run ids must match `^pb-\d{8}-[a-z0-9]{6}$`; every artifact path is resolved and asserted to stay inside the runs directory; cache file names are SHA-1 of the URL. | `security.py::validate_run_id`, `security.py::safe_child` |
| **Half-written outputs** | `company_brain.json`, `evidence.json`, `report.md` are validated first and written atomically (temp file + `os.replace`). Run databases and logs are created `0600`. A fatal run never writes a profile. | `security.py::atomic_write_text`, `workflow/export.py` |
| **Terminal injection** from scraped titles or answers | All untrusted text is escaped for Rich markup and stripped of ANSI/control characters before printing; interview answers are capped at 2 000 characters. | `security.py::safe_text`, `security.py::sanitize_answer` |
| **Resource exhaustion** | Timeouts on every network call (`Firecrawl timeout`, `ChatOpenAI(timeout=60, max_retries=0)`), per-page character cap, candidate cap, page/question/scrape-call/model-call caps, optional USD budget, retries capped at `MAX_RETRIES` with bounded backoff. Firecrawl's internal retry loop is disabled so attempts stay bounded. | `config.py`, `agent/middleware.py`, `web/scraper.py` |
| **SQL injection / data integrity** | Parameterized queries only; JSON columns written with `json.dumps`; WAL mode; schema version recorded. | `state/run_store.py` |
| **Supply chain** | `uv.lock` pins every dependency; CI runs `pip-audit` and `ruff` with the `S` (bandit) rule set. | `pyproject.toml`, `.github/workflows/ci.yml` |
| **Polite crawling** | `robots.txt` respected (per-host cache), identifying `User-Agent`, one tool call per step so there are no fetch storms. | `web/robots.py`, `agent/middleware.py::SerialToolCallsMiddleware` |

## Reporting

This is a take-home project; open an issue in the repository if you find a problem.
