# Container architecture

## Scope

The local container stack packages the existing FastAPI and React product
without changing financial or ML responsibilities. Portfolio calculations stay
inside deterministic backend services, the browser consumes API results, and
the category model remains a versioned backend artifact.

The stack is intended to provide a reproducible development and deployment
baseline. It does not provision cloud infrastructure or introduce production
claims for the experimental classifier.

## Service topology

```text
Browser
  |
  | http://localhost:5173
  v
web: unprivileged Nginx on port 8080
  |-- / and static assets -> React production build
  |-- /health            -> Nginx health response
  `-- /api/*             -> http://api:8000/*
                                |
                                v
                         api: FastAPI on port 8000
                                |
                                v
                    financial_ai_runtime volume
```

Compose publishes the web container's port `8080` as host port `5173` and the
API container's port `8000` as host port `8000`. The direct API mapping keeps
OpenAPI documentation and diagnostics accessible during local development. Web
application requests use `/api`, so browser traffic remains same-origin and
Nginx performs the internal service lookup.

Compose creates a private default network. The service name `api` is an
internal DNS name on that network; it is not a hostname exposed to the browser.

## API image

`docker/api/Dockerfile` starts from `python:3.12-slim` and copies a pinned `uv`
binary from the official distroless uv image. Dependency installation is split
into two layers:

1. `uv sync --locked --no-dev --no-install-project` installs locked third-party
   runtime dependencies before frequently changing source files are copied.
2. `uv sync --locked --no-dev --no-editable` installs the application as a
   non-editable package after the source is present.

The runtime process uses a dedicated unprivileged `financial-ai` user. Only
`/app/data/runtime` is writable. On startup the container:

1. applies Alembic migrations;
2. creates the bilingual category model only when either required model file is
   absent;
3. replaces the startup shell with Uvicorn listening on `0.0.0.0:8000`.

Conditional model initialization prevents a container restart from
overwriting a previously promoted model stored in the runtime volume.

## Web image

`docker/web/Dockerfile` is a multi-stage build:

1. `node:22-alpine` installs the exact npm lockfile and creates the Vite
   production bundle.
2. an unprivileged Nginx image receives only `dist/` and the reviewed server
   configuration.

Node.js, TypeScript, source files, and `node_modules` are therefore absent from
the final web image. Nginx provides the React single-page-application fallback,
the `/api` reverse proxy, a lightweight health endpoint, and baseline response
headers.

Local Vite development uses the same `/api` browser path and rewrites it to
`http://localhost:8000`. Container traffic uses the Nginx proxy instead. This
keeps API calls environment-independent from the React application's point of
view.

## Persistence

The named `financial_ai_runtime` volume is mounted at `/app/data/runtime` and
persists independently of an API container. It contains local runtime state,
including:

- the SQLite database;
- generated training snapshots and the active category model;
- exported feedback snapshots;
- candidate artifacts, evaluation reports, archives, and promotion receipts.

`docker compose down` removes containers and the Compose network but retains
this volume. `docker compose down --volumes` deliberately and permanently
removes the container-managed runtime state.

The repository-level `.dockerignore` excludes local environments, dependency
trees, build outputs, editor settings, Git metadata, runtime data, databases,
and nested `.env` files from both image build contexts.

## Readiness and lifecycle

The API healthcheck calls `/health` with Python's standard library, avoiding an
extra operating-system package solely for probing. Its start period allows for
the initial migration and model build. The web service starts only after the
API reports healthy and exposes its own Nginx `/health` endpoint.

Both services enable Compose's minimal init process for signal and child-process
handling. Uvicorn becomes the API container's final process through `exec`, so
container stop signals reach it directly.

## Continuous integration

The `Containers` CI job runs independently from unit-level backend and frontend
jobs. It:

1. validates the normalized Compose configuration;
2. builds both images from their Dockerfiles;
3. starts the stack and waits for declared healthchecks;
4. probes FastAPI directly, Nginx directly, and FastAPI through the Nginx proxy;
5. prints container logs on failure;
6. removes containers, the network, and the temporary CI volume on every run.

The proxy smoke test exercises the complete browser-facing path rather than
only confirming that two isolated processes started.

## Current boundaries

- SQLite is appropriate for this single-instance local release; horizontally
  scaled deployment requires an external database and a migration strategy.
- The generated model and SQLite state share one local volume for simplicity.
  A deployed architecture should separate database, model registry, and durable
  feedback storage concerns.
- TLS termination, authentication, secrets management, image publication,
  vulnerability scanning, and cloud orchestration remain future deployment
  work.
