# Model retirement

Retirement is a governed lifecycle action, not simply deletion of a model file.

## Triggers

Retirement should be considered when:

- performance or calibration remains inadequate after reasonable remediation;
- sustained drift places the population outside the validated scope;
- fairness or proxy risks cannot be reduced adequately;
- a prohibited use has become embedded in operations;
- data provenance or lawful-use assumptions can no longer be defended;
- privacy or security risk cannot be controlled;
- the business purpose changes materially;
- a successor model has been approved;
- the supporting service is discontinued.

## Process

### Freeze and identify scope

Record the model version, fingerprint, threshold, active endpoints, dashboards, scheduled jobs, dependent workflows and approved uses.

### Stop new use

Disable active scoring paths, scheduled jobs and routing to the retired model. A replacement should be activated only after its own approval.

### Preserve accountability evidence

Retain the minimum evidence needed to reconstruct the lifecycle: source commit, configuration, model card, validation results, monitoring summaries, governance decisions and incident records.

Unnecessary row-level personal data should not be retained for this purpose.

### Revoke obsolete access

Remove credentials, deployment hooks, scheduled jobs, service permissions and storage access that exist solely for the retired model.

### Communicate

Affected users should know when the model is retired, why its use stopped, whether a replacement exists and how historical scores should be treated.

### Verify closure

Confirm that the model is no longer reachable through active deployment paths and that remaining evidence has a defined retention purpose.
