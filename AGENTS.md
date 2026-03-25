# Credit Risk Analytics Project: Engineering Guidelines & Architecture

This document outlines the engineering standards, architecture, and operational procedures for the Credit Risk Analytics project. It is intended for both human developers and AI agents.

---

## 1. Operations & Tooling

### 1.1 Development Environment
Ensure your environment is set up by installing the requirements:
```bash
pip install -r requirements.txt
```

### 1.2 Build & Verification Commands
Always run these before submitting any code changes.

- **Linting & Formatting:**
  ```bash
  ruff check . --fix
  ruff format .
  ```
- **Type Checking:**
  ```bash
  mypy src
  ```
- **Testing:**
  ```bash
  pytest
  ```

---

## 2. Code Style & Engineering Standards

### 2.1 General Principles
- **PEP-8 Compliance:** All Python code must strictly follow PEP-8 conventions.
- **Type Hints:** Mandatory for all function signatures and public class attributes. Use `typing` module for complex types.
- **Documentation:** Use Google-style docstrings for all modules, classes, and public functions.

### 2.2 Error Handling
- Never use bare `except:` blocks.
- Prefer specific exceptions (e.g., `ValueError`, `FileNotFoundError`) over generic `Exception`.
- Custom exceptions should inherit from a project-level `CreditRiskError`.
- Log exceptions with context using the standard `logging` library.

### 2.3 Functional vs. OOP
- Use **Pure Functions** for data transformations to ensure testability and reproducibility.
- Use **Classes** for managing stateful entities like ML models, database connections, or API clients.

---

## 3. Data Architecture Overview

The system follows a modular data pipeline architecture designed for scalability, observability, and auditability.

### 3.1 Data Ingestion Layer
- **Source:** External credit bureaus, internal transaction logs, and user-provided application data.
- **Process:** Raw data is ingested into an Amazon S3 landing zone (Bronze Layer) in its native format (JSON/CSV).
- **Validation:** Schema validation occurs at the entry point to prevent downstream corruption.

### 3.2 Transformation Layer (Silver)
- **Cleaning:** Handling missing values, outlier detection, and normalization.
- **Feature Engineering:** Calculation of credit scores, debt-to-income ratios, and behavioral metrics.
- **Storage:** Structured Parquet files or Delta tables for optimized query performance.

### 3.3 ML Modeling Layer (Gold)
- **Training:** LightGBM and Scikit-learn pipelines for credit default prediction.
- **Experiment Tracking:** MLflow is used to track hyperparameters, metrics, and model artifacts.
- **Inference:** Models are served via a REST API (FastAPI) or batch scoring jobs.

### 3.4 BI & Analytics Layer
- **Reporting:** Aggregated metrics exported to Snowflake or BigQuery.
- **Visualization:** Dashboards in PowerBI or Streamlit for risk monitoring and executive reporting.

---

## 4. Testing Strategy

### 4.1 Unit Tests
- Location: `tests/unit/`
- Goal: Test individual functions and logic in isolation. Mock all external dependencies.

### 4.2 Integration Tests
- Location: `tests/integration/`
- Goal: Verify the interaction between internal components (e.g., Data Loader -> Transformer).

### 4.3 Pipeline Tests
- End-to-end verification of the data flow from ingestion to model inference.

---

## 5. Security & Compliance

- **PII Handling:** No Personally Identifiable Information should be logged or stored in cleartext.
- **Secrets Management:** Use environment variables (via `.env`) or AWS Secrets Manager. NEVER commit secrets.
- **Audit Trails:** Every model decision must be logged with the input data version and model version for regulatory compliance.

---

## 6. Git Workflow

- **Branching:** Use `feature/`, `bugfix/`, or `refactor/` prefixes.
- **Commits:** Conventional Commits (e.g., `feat: add debt-to-income calculation`).
- **PRs:** Must pass all linting and test checks before review.

---

## 7. Performance Optimization

- Use `pandas` vectorization; avoid `iterrows()` or manual loops.
- For large datasets, consider `dask` or `polars` for parallel processing.
- Profile bottlenecks using `cProfile` or `line_profiler`.

---

*This guide is a living document. Updates must be approved by the Lead Engineer.*

---
*End of Document*
