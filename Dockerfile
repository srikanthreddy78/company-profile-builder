FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.12.13 \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
# Dependencies first (cached until pyproject.toml / uv.lock change), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" PROFILE_BUILDER_RUNS_DIR=/runs
RUN mkdir -p /runs && chown 10001:10001 /runs
VOLUME ["/runs"]
USER 10001
ENTRYPOINT ["python", "-m", "profile_builder"]
CMD ["--help"]
