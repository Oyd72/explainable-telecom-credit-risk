# Explainable telecom credit risk

Explainable and monitorable short-horizon repayment risk modelling using public telecom microcredit data.

## Project purpose

This project explores how short-horizon repayment risk can be estimated from behavioural transaction data in a way that remains explainable, reproducible and monitorable over time. The model is designed to support prioritisation of higher-risk cases for follow-up, not to make automated lending decisions. The project also examines how temporal drift, calibration and governance affect whether a model that performs well statistically remains suitable for operational use.

## Data

The project uses a publicly available telecom microcredit dataset from Kaggle. Raw customer-level data are not stored in this repository. Reproduction instructions will explain how to obtain the source dataset and place it in the expected local directory before running the pipeline.

The unexplained change in outcome labels after 23 July 2016 is treated as a data-quality and temporal-validity issue, not as ordinary supervised ground truth.

## Repository structure

The repository is organised around data preparation, modelling, explainability, monitoring, inference, deployment and governance. The initial structure is being created before selected components are rebuilt from the earlier prototype.
