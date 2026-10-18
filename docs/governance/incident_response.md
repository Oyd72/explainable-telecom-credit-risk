# Model incident response

A model incident is an event that may make model use unreliable, harmful, unauthorised or inconsistent with the approved purpose.

## Examples

Incidents may include:

- material data-quality failure;
- unexpected performance deterioration;
- serious drift outside the validated population;
- use of the model for an unapproved automated decision;
- discovery of a plausible discriminatory effect;
- privacy or security exposure;
- corrupted or mismatched model artefacts;
- missing lineage or inability to reproduce a deployed score.

## Response process

### Detect and record

Create an incident record with the date, model version, affected use, evidence and initial owner.

### Assess severity

Consider potential customer impact, breadth of affected data, reliability of current scores, legal or privacy implications and whether the issue is ongoing.

### Contain

Possible containment measures include suspending scoring, disabling a deployment path, restricting a use case, reverting to a known model version or requiring manual handling.

### Investigate

Preserve relevant aggregate evidence and determine whether the issue originates in source data, transformation logic, model behaviour, calibration, threshold choice, deployment or downstream use.

### Remediate

Corrections may involve data repair, code changes, recalibration, retraining, feature restriction, process changes or additional human review.

A new model should not be promoted merely because it appears to resolve the incident; it must pass the normal validation and approval process.

### Resume or retire

Resume use only when the responsible owner accepts the residual risk and the control failure has been addressed. If the issue cannot be reduced adequately, use should remain suspended or the model should be retired.

## Historical impact

Where an incident could have affected consequential past decisions, the deployment owner should consider whether historical cases need review. The scope depends on the nature of the failure and the operational context.
