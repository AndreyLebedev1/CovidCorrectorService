FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser main.py model.py covid_corrector/ bin/ ./
RUN chown -R appuser:appuser /app/.venv

USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
