# Telco Support Agent

Production-grade AI-powered customer support Agent System built on Databricks.

## Overview

This project implements a multi-agent system that processes telecom customer support queries in real-time, retrieves relevant information, and provides human agents with contextually appropriate responses.

## Features

- Multi-agent supervisor orchestrating specialized sub-agents
- Routing cache with similarity-based lookup to reduce LLM calls
- Vector Search for unstructured data retrieval
- Tool-calling agents for structured data retrieval
- Agent lifecycle tracked using MLflow and Unity Catalog
- Production monitoring using Agent Evaluation and MLflow Tracing
- Lakehouse App for front-end user interface

## Repo Structure

```
agentic-customer-support/
├── README.md
├── pyproject.toml           # uv for dependency management
├── uv.lock                  # uv lock file
├── requirements.txt         # Dependencies for notebook installation
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── databricks.yml           # Databricks Asset Bundle config
│
├── resources/               # Databricks Asset Bundle resources
│   ├── data_pipelines/      # Data pipeline resources
│   ├── agent/               # Agent resources
│   └── ui/                  # UI app resources
│
├── configs/                 # Configuration files
│   ├── agents/              # Agent configurations
│   └── data/                # Data pipeline configurations
│
├── .github/                 # CI/CD workflows
│   └── workflows/
│       ├── pr-validation.yml           # PR validation
│       ├── dev-deploy-resources.yml    # Dev deployment
│       ├── staging-deploy-resources.yml # Staging deployment
│       └── prod-deploy-resources.yml   # Production deployment
│
├── telco_support_agent/
│   ├── __init__.py
│   ├── config/              # Config management
│   │
│   ├── agents/              # Agent implementations
│   │   ├── __init__.py
│   │   ├── supervisor.py    # Supervisor agent
│   │   ├── account.py       # Account management agent
│   │   ├── billing.py       # Billing agent
│   │   ├── tech_support.py  # Tech support agent
│   │   └── product.py       # Product info agent
│   │
│   ├── cache/               # Routing cache management
│   │   └── manager.py       # Cache manager with similarity-based lookup
│   │
│   ├── tools/               # Agent tools
│   │
│   ├── data/                # Data generation and management
│   │
│   ├── evaluation/          # Evaluation framework
│   │
│   ├── mcp_servers/         # MCP server implementations
│   │   └── outage_info_server/
│   │
│   ├── ops/                 # AgentOps utilities
│   │
│   └── ui/                  # Customer support UI application
│
├── notebooks/               # Databricks notebooks
│   ├── 00_data_generation/  # Synthetic data generation
│   ├── 01_vector_search_setup/ # Vector search index creation
│   ├── 02_run_agent/        # Agent testing and development
│   ├── 03_log_register_agent/ # MLflow logging and UC registration
│   ├── 04_evals/            # Agent evals
│   └── 05_deploy_agent/     # Model serving deployment
│
├── scripts/                 # Utility scripts
│   ├── lint.sh              # Linting script
│   └── run_tech_support_agent.py  # Tech support agent runner
│
└── tests/                   # Unit tests
    ├── conftest.py
    └── test_data/
```

## Development Environment

This project is designed to run on [Databricks Runtime 16.3](https://docs.databricks.com/aws/en/release-notes/runtime/16.3), which uses **Python 3.12.3**. For consistency between development and production environments, we recommend using the same Python version locally.

### Environment Setup

#### Prerequisites
- Python 3.12+
- uv for dependency management

#### Local Development

For local development, create a virtual environment using uv:

```bash
# Install Python 3.12.3
pyenv install 3.12.3
pyenv local 3.12.3

# Check Python version
python --version  # Should output Python 3.12.3

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Managing Dependencies

#### Adding or Updating Dependencies

When adding new dependencies or updating existing versions:

1. Update `pyproject.toml` - add or modify dependencies in the `pyproject.toml` file
2. Update the lock file - run `uv lock` to update the `uv.lock` file
3. Regenerate `requirements.txt` - run `uv export --format requirements-txt > requirements.txt`
4. Sync your environment - run `uv sync` to update your local environment

```bash
# Example: adding a new dependency
uv add new-package

# Example: updating a specific package
uv add package-name==2.0.0

# after any dependency changes:
uv lock
python scripts/generate_requirements.py
uv sync
```

**Important**: The `requirements.txt` file contains only the direct dependencies from `pyproject.toml` with semantic versioning ranges. This format is optimized for Databricks notebook installations and avoids conflicts with pre-installed packages. Use the `scripts/generate_requirements.py` script to regenerate it after any dependency changes.

### Using requirements.txt

We maintain a `requirements.txt` file for pip installing dependencies in notebooks. The file is kept in sync with the `pyproject.toml` dependencies and can be installed directly:

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### Databricks Development

When running on Databricks:

1. Use Databricks Runtime 16.3
2. Install project-specific dependencies at the start of your notebook:

```python
%pip install -r /path/to/project/requirements.txt
```

## Code Quality & Linting

This project uses Ruff for linting and formatting, mypy for type checking, and pre-commit hooks for automating quality checks.

### Setting Up

```bash
# install development dependencies
uv sync

# set up pre-commit hooks
# NOTE: may run into issues with this if also running Databricks pre-commit git hooks
uv run pre-commit install
```

### Running Linting Manually

```bash
# use the provided script
chmod +x scripts/lint.sh
./scripts/lint.sh

# or run commands individually
uv run ruff format .                  # Format code
uv run ruff check . --fix             # Run linter with auto-fixes
uv run mypy telco_support_agent       # Type checking
```

### Code Style

The project follows these standards:
- Google-style docstrings
- 88-character line length (Black-compatible)
- Double quotes for strings
- Type annotations for all functions
- Pre-commit hooks enforce these standards automatically

## Data Generation

The project includes a synthetic data generation module that creates realistic telecom customer data:

- Customer profiles and subscriptions
- Plans, devices, and promotions
- Billing and usage records
- Knowledge base articles and support tickets

## Testing

TODO

## Deployment

The project uses Databricks Asset Bundles (DAB) for deployment across three environments: dev, staging, and prod.

### CI/CD Pipeline

1. **PR Validation** (`pr-validation.yml`)
   - Runs on all PRs to main branch
   - Performs linting (Ruff), formatting checks, and unit tests
   - Validates Databricks Asset Bundle configurations for all environments

2. **Dev Deployment** (`dev-deploy-resources.yml`)
   - Manual workflow dispatch
   - Deploys to dev environment
   - Always deploys MCP server
   - UI deployment:
     - Currently uses `deploy.sh` script as a temporary workaround (builds frontend assets properly)
     - Automatic: deploys when changes detected in `telco_support_agent/ui/` or `resources/ui/customer_support_app.yml`
     - Manual: force deploy with `deploy_ui` flag
   - Runs on protected runner group

3. **Staging Deployment** (`staging-deploy-resources.yml`)
   - Automatically triggers on push to main branch
   - Manual workflow dispatch available
   - Always deploys MCP server
   - UI deployment:
     - Currently uses `deploy.sh` script as a temporary workaround (builds frontend assets properly)
     - Automatic: deploys when changes detected in `telco_support_agent/ui/` or `resources/ui/customer_support_app.yml`
     - Manual: force deploy with `deploy_ui` flag
   - Uses Python 3.10 for compatibility

4. **Production Deployment** (`prod-deploy-resources.yml`)
   - Triggers on release publication or manual dispatch
   - Requires semantic versioning tag (e.g., v1.2.3)
   - Validates prerequisites:
     - Tag must be on main branch
     - Staging deployment must be successful
   - Always deploys MCP server
   - UI deployment:
     - Currently uses `deploy.sh` script as a temporary workaround (builds frontend assets properly)
     - Automatic: deploys when changes detected in `telco_support_agent/ui/` or `resources/ui/customer_support_app.yml`
     - Manual: force deploy with `deploy_ui` flag
   - Uses Python 3.10 for compatibility

### Manual Deployment Commands

```bash
# Deploy to dev environment
databricks bundle deploy -t dev

# Deploy and run specific resources
databricks bundle run log_register_deploy_agent -t dev
databricks bundle run create_vector_indexes -t dev
databricks bundle run mcp_telco_outage_info -t dev

# UI deployment (currently using deploy.sh script)
cd telco_support_agent/ui
./deploy.sh dev  # or staging/prod

# Deploy to staging/prod (similar commands with -t staging or -t prod)
```

**Note**: UI deployment currently uses the `deploy.sh` script as a temporary workaround because the bundle sync approach doesn't properly build frontend assets. The script handles frontend building, static file management, and deployment to the correct environment.

### Databricks Resources

The project deploys the following resources via Asset Bundles:

1. **Agent Resources** (`resources/agent/`)
   - `log_register_deploy_agent`: MLflow logging and model serving deployment
   - `mcp_telco_outage_info`: MCP server for outage information

2. **Data Pipeline Resources** (`resources/data_pipelines/`)
   - `create_vector_indexes`: Creates vector search indexes for knowledge base and support tickets

3. **UI Resources** (`resources/ui/`)
   - `customer_support_ui`: Customer support application interface
