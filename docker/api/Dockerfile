FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project --extra semantic

COPY README.md alembic.ini ./
COPY apps/api/src ./apps/api/src
COPY apps/api/alembic ./apps/api/alembic
COPY data/market ./data/market
COPY data/development ./data/development

RUN uv sync --locked --no-dev --no-editable --extra semantic

ENV PATH="/app/.venv/bin:$PATH"

RUN groupadd --system financial-ai \
    && useradd --system --gid financial-ai --no-create-home financial-ai \
    && mkdir -p /app/data/runtime \
    && chown financial-ai:financial-ai /app/data/runtime

ENV FINANCIAL_AI_DATABASE_URL="sqlite:////app/data/runtime/financial_ai.db" \
    FINANCIAL_AI_ECB_FX_PATH="/app/data/market/ecb_fx.csv" \
    FINANCIAL_AI_CATEGORY_MODEL_ARTIFACT_PATH="/app/data/runtime/ml/models/transaction_category_bilingual_v1.pkl" \
    FINANCIAL_AI_CATEGORY_MODEL_METADATA_PATH="/app/data/runtime/ml/models/transaction_category_bilingual_v1.json"

USER financial-ai

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && financial-ai-bootstrap-category-model && exec uvicorn financial_ai.main:app --host 0.0.0.0 --port 8000"]
