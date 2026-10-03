FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir uv==0.12.13
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" PROFILE_BUILDER_RUNS_DIR=/runs
VOLUME ["/runs"]
ENTRYPOINT ["python", "-m", "profile_builder"]
CMD ["--help"]
