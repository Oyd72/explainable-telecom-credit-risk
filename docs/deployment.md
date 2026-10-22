# Deployment

The project can be published without committing raw source data or a model binary.

## Streamlit Community Cloud

Two applications are ready for deployment directly from the `main` branch.

### Stakeholder dashboard

- repository: `Oyd72/explainable-telecom-credit-risk`
- branch: `main`
- main file: `dashboards/stakeholder_app.py`

### Monitoring dashboard

- repository: `Oyd72/explainable-telecom-credit-risk`
- branch: `main`
- main file: `dashboards/monitoring_app.py`

The dashboard-specific dependency file is `dashboards/requirements.txt`.

Both dashboards use aggregate evidence committed under `reports/`; they do not require the raw dataset or model binary.

## Render API

The repository includes a Render Blueprint in `render.yaml`.

During the Render build:

1. serving dependencies are installed;
2. `scripts/build_deployment_model.py` downloads the public Kaggle dataset;
3. the repository cleaning and feature logic is applied;
4. the Random Forest and isotonic calibrator are rebuilt;
5. the generated model artefact is written locally inside the deployment;
6. temporary row-level data are deleted;
7. FastAPI starts with Uvicorn.

The public source data and generated model binary remain outside Git version control.

Expected API routes:

- `/`
- `/health`
- `/docs`
- `/predict`

If Kaggle changes access requirements for the public dataset, Render may need a `KAGGLE_API_TOKEN` environment variable. The current KaggleHub documentation states that authentication is only required for public resources that require user consent or for private resources.
