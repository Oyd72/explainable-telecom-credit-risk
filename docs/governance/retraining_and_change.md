# Retraining and model change

Retraining is treated as a controlled model change, not routine background maintenance.

## Trigger

A retraining review may be opened when:

- reliable new labelled data accumulate;
- drift remains material;
- performance or calibration deteriorates;
- a material feature or business-process change occurs;
- an incident reveals a model weakness.

Drift alone does not automatically justify retraining.

## Candidate process

The implementation in `src/modeling/retrain.py`:

1. checks the feature contract and labelled batch size;
2. splits the later batch chronologically into update and validation portions;
3. combines the update portion with trusted historical development data;
4. reserves a recent seven-day calibration slice;
5. retrains the same Random Forest architecture;
6. recalibrates with isotonic regression;
7. evaluates the candidate on the untouched later validation portion;
8. applies configured quality gates;
9. writes a candidate artefact and summary.

## Promotion

Passing the quality gates is necessary but not sufficient for replacement.

Automatic promotion is disabled. Human review should consider:

- predictive performance;
- calibration;
- drift and stability;
- fairness evidence and limitations;
- data-quality changes;
- explanation behaviour;
- incident history;
- operational consequences of changing the model.

The currently deployed version remains unchanged until an explicit promotion decision is made.

## Demonstration limitation

The repository can demonstrate this mechanism using the historical final holdout as a labelled batch. That proves the workflow runs, but it is not equivalent to genuinely new post-deployment evidence.
