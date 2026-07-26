# Financial AI Assistant

Local-first portfolio intelligence platform with deterministic analytics, a
FastAPI API, and a React dashboard. Phase 1 intentionally uses synthetic market
prices and bundled ECB-style FX fixtures so the demo is reproducible and never
mistaken for live financial data.

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 22 or newer and npm

## Run locally

```powershell
python -m uv sync --all-groups
python -m uv run alembic upgrade head
python -m uv run financial-ai-api
```

In a second terminal:

```powershell
cd apps/web
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173`. API documentation is available at
`http://localhost:8000/docs`.

## Tests

```powershell
python -m uv run pytest
cd apps/web
npm.cmd test
npm.cmd run build
```

`python -m uv` also works when the `uv` executable is not yet available on the
current terminal's `PATH`. Use `npm.cmd` in PowerShell environments where the
execution policy blocks `npm.ps1`. Generated local content in `.venv`,
`node_modules`, `dist`, and `data/runtime` is excluded from version control.

The application is educational software, not financial advice. All included
security prices are synthetic demo data.
