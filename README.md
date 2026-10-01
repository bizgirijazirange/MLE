# MLE Supply Chain Pipeline — Declarative Automation Bundle (DAB)

## What is a Declarative Automation Bundle (DAB)?

A **DAB** (formerly "Databricks Asset Bundle") is Databricks' source-control-first
framework for defining, deploying, and managing data and ML pipelines as code.

Instead of manually configuring jobs in the Databricks UI, you describe your
infrastructure — jobs, pipelines, notebooks, variables, and environment-specific
settings — in a `databricks.yaml` file at the root of your Git repository.
DAB handles syncing source code to the workspace, resolving environment-specific
variables, and creating/updating Databricks resources on every deploy.

### Key benefits

* **Infrastructure as code** — every job, cluster, and variable lives in YAML,
  version-controlled alongside your Python source.
* **Environment targets** — define `dev`, `staging`, `prod` targets with different
  workspace paths, variables, and permissions in a single file.
* **Git-based CI/CD** — push to `main` and a GitHub Actions workflow automatically
  validates, deploys, and runs the pipeline. No manual UI steps.
* **Idempotent deploys** — `databricks bundle deploy` creates or updates the job
  in-place; re-running it does not create duplicates.
* **Variable substitution** — use `${var.config_file}`, `${workspace.root_path}`,
  `${bundle.name}`, `${workspace.current_user.name}` to keep configs DRY.

---

## Project Structure

```
MLE/                         ← Git repo root (DAB bundle root)
├── databricks.yaml          ← DAB bundle definition (this file is the "source of truth")
├── README.md                ← This file
├── requirements.txt          ← Python dependencies (for reference)
├── .gitignore               ← Git ignore patterns (also respected by DAB sync)
├── config/
│   └── dev.yaml             ← Environment config (catalog, schemas, pipeline settings)
├── src/
│   ├── main.py              ← Orchestrator entry point (run by the DAB job)
│   ├── meadallion_pipeline.py  ← DE classes: BronzeIngestion, SilverIngestion, GoldIngestion
│   ├── ml_pipeline.py       ← ML classes: FeatureEngineering, ModelTraining
│   └── exploratory_data_analysis  ← EDA notebook
├── .github/
│   └── workflows/
│       └── cd.yaml          ← GitHub Actions CI/CD pipeline
└── .bundle/                 ← (generated on deploy — not committed)
```

### Execution flow

```
main.py
  │
  ├── Stage 1: Data Engineering
  │     ├── BronzeIngestion.run()   — CSV → Bronze Delta table
  │     ├── SilverIngestion.run()   — Bronze → Silver (dedup + DQ + quarantine)
  │     └── GoldIngestion.run()     — Silver → Gold (aggregated to product-city-date grain)
  │
  └── Stage 2: Machine Learning
        ├── FeatureEngineering.run()  — Gold → Feature Store (lag, rolling, calendar features)
        └── ModelTraining.run()       — Train → Register → Predict (LinearRegression + MLflow)
```

---

## How `databricks.yaml` Works

### Variables

```yaml
variables:
  config_file:
    description: Workspace path to the environment config YAML file
    default: ""
```

The `config_file` variable is set per target (see below).  `main.py` receives it
as a `--config-file` command-line argument via the `spark_python_task.parameters`
field.  No Databricks widgets are used.

### Job definition

```yaml
resources:
  jobs:
    mle_pipeline_job:
      tasks:
        - task_key: run_pipeline
          spark_python_task:
            python_file: ${workspace.root_path}/src/main.py
            parameters:
              - --config-file
              - ${var.config_file}
          libraries:
            - pypi:
                package: databricks-feature-engineering
```

* `python_file` — resolved to the workspace-synced path of `src/main.py`.
* `parameters` — passed as `sys.argv` to `main.py`; `argparse` reads `--config-file`.
* `libraries` — pip packages installed on the job cluster at run time.

### Targets

```yaml
targets:
  dev:
    mode: development
    workspace:
      root_path: /Workspace/Users/${workspace.current_user.name}/.bundle/${bundle.name}
    variables:
      config_file: ${workspace.root_path}/config/dev.yaml

  prod:
    mode: production
    workspace:
      root_path: /Workspace/Shared/.bundle/${bundle.name}
    variables:
      config_file: ${workspace.root_path}/config/prod.yaml
```

Each target:

* `mode: development` — allows local overrides; `mode: production` enforces stricter rules.
* `workspace.root_path` — where DAB syncs your source files in the Databricks workspace.
* `variables.config_file` — the workspace path to the YAML config for that environment.

---

## Prerequisites

### 1. Install the Databricks CLI (v0.200+)

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/databricks/cli/main/install.sh | sh

# Verify
databricks --version
```

### 2. Authenticate

```bash
# Option A: interactive OAuth (recommended for local dev)
databricks auth login --host https://<your-workspace>.cloud.databricks.com

# Option B: personal access token (PAT)
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
export DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXX
```

### 3. Create a prod config (optional)

For the `prod` target, create `config/prod.yaml` with production-specific settings
(catalog, schemas, source_path, model thresholds).  You can copy `dev.yaml` as a
starting point and adjust the values.

---

## Local Development Workflow

```bash
# 1. Validate the bundle (checks YAML syntax, variable resolution, etc.)
databricks bundle validate --target dev

# 2. Deploy to the dev workspace (creates/updates the Databricks Job)
databricks bundle deploy --target dev

# 3. Run the job manually
databricks bundle run mle_pipeline_job --target dev

# 4. Check job status in the Databricks UI or via CLI
databricks jobs list --output JSON
```

---

## Git-Based Deployment (CI/CD)

The `.github/workflows/cd.yaml` GitHub Actions workflow automates the entire
deployment process:

### Flow

```
Pull Request → main      →  databricks bundle validate --target dev   (dry-run only)
Push to main              →  databricks bundle validate --target dev
                          →  databricks bundle deploy   --target dev
                          →  databricks bundle run mle_pipeline_job --target dev
Manual trigger            →  same as push to main
```

### Required GitHub Secrets

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** →
**New repository secret** and add:

| Secret Name | Value |
| --- | --- |
| `DATABRICKS_HOST` | Your workspace URL (e.g. `https://<workspace>.cloud.databricks.com`) |
| `DATABRICKS_TOKEN` | A Databricks personal access token (PAT) or service-principal token |

### Steps to enable

1. Commit all files (`databricks.yaml`, `src/main.py`, `.github/workflows/cd.yaml`, etc.) to your Git repo.
2. Push to the `main` branch (or open a PR).
3. GitHub Actions will automatically pick up the workflow.
4. Check the **Actions** tab in GitHub to monitor the pipeline.

### Branching strategy (recommended)

```
feature/*  →  PR to main  →  validate (dry-run)
main       →  push         →  deploy to dev target
tag v*.*.* →  (future)     →  deploy to prod target
```

---

## DAB Commands Cheat Sheet

| Command | Description |
| --- | --- |
| `databricks bundle validate --target dev` | Validate bundle config (syntax, variables, permissions) |
| `databricks bundle deploy --target dev` | Sync files + create/update Databricks resources |
| `databricks bundle run mle_pipeline_job --target dev` | Trigger the job immediately |
| `databricks bundle summary` | Show bundle overview (resources, variables, targets) |
| `databricks bundle destroy --target dev` | Remove all resources created by the bundle |

---

## Notes

* **No widgets** — `main.py` uses `argparse` for the `--config-file` parameter instead of Databricks widgets. The DAB job passes the config path as a `spark_python_task` parameter.
* **Serverless compute** — the job runs on serverless compute by default (no cluster config needed). Libraries are installed via the `libraries` section.
* **Idempotent** — re-deploying the same target updates the existing job; it does not create a duplicate.
* **Sync exclusions** — `.github/` and `demand_forecasting/` are excluded from workspace sync via the `sync.exclude` section.
* **Schedule** — the job includes a daily 08:00 UTC schedule, but it is **paused** by default. Unpause it in the Databricks Jobs UI or set `pause_status: UNPAUSED` in `databricks.yaml`.