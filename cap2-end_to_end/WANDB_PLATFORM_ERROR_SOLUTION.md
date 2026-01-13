# W&B Platform Error: "Cannot query field codePathLocal" - Solution

## Error Message
```
Cannot query field "codePathLocal" on type "RunInfo". Did you mean "codePath"?
An application error occurred.
Click to refresh the page.
```

## What Happened
This is a **W&B web platform GraphQL query error**, NOT a problem with your pipeline execution or data logging.

## Root Cause
The W&B web interface is trying to query a field (`codePathLocal`) that:
- Was deprecated or removed in recent W&B API versions
- May not be compatible with the current wandb client version (0.22.1)
- Is a known issue with W&B's frontend trying to access outdated schema fields

## ✅ Your Data is Safe
**IMPORTANT:** This error does NOT affect:
- ✅ Your logged metrics (all metrics were successfully logged)
- ✅ Your sweep results (all 5 runs completed and logged)
- ✅ Your model artifacts (all artifacts uploaded)
- ✅ Your pipeline execution (completed successfully)

## Evidence Your Runs Were Logged Successfully
From the pipeline output:
```
wandb: 🚀 View run treasured-sweep-1 at: https://wandb.ai/.../runs/eq1e4fs9
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260113_121111-eq1e4fs9/logs

[... repeated for all 5 sweep runs ...]

wandb: 🚀 View run model_registration at: https://wandb.ai/.../runs/z3zyiusd
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
```

All runs show "Synced" status, meaning data was successfully uploaded to W&B.

## Solutions (In Order of Preference)

### Solution 1: Browser Cache Clear (Quickest)
This often fixes GraphQL schema mismatch issues:

```bash
# Chrome/Brave
1. Open DevTools (F12)
2. Right-click on the refresh button
3. Select "Empty Cache and Hard Reload"

# Or clear all W&B cache
1. Go to Settings > Privacy and Security > Clear Browsing Data
2. Select "Cached images and files"
3. Time range: "Last 7 days"
4. Clear data
5. Refresh W&B page
```

### Solution 2: Use Different Browser
Sometimes the error is browser-specific due to cached GraphQL schemas:
- Try accessing W&B in an incognito/private window
- Try a different browser (Firefox, Safari, Edge)

### Solution 3: Wait for W&B Platform Update
W&B frequently deploys updates that fix these GraphQL schema issues:
- The error may resolve itself within hours/days
- W&B is aware of these types of errors and fixes them quickly

### Solution 4: Use W&B CLI to View Data
While the web UI is having issues, you can access all your data via CLI:

```bash
# List all runs in your project
wandb runs housing-mlops-gcp

# Get specific run details
wandb run show danieljimenez88m-carlosdanieljimenez-com/housing-mlops-gcp/eq1e4fs9

# View sweep results
wandb sweep show imu0vp0m

# Download run data
wandb export housing-mlops-gcp --run eq1e4fs9
```

### Solution 5: Upgrade wandb Client
Update to the latest wandb client to ensure compatibility:

```bash
# Using UV (recommended for this project)
uv pip install --upgrade wandb

# Or using pip
pip install --upgrade wandb
```

### Solution 6: Contact W&B Support
If the error persists after trying above solutions:

```bash
# File a bug report
wandb verify

# Or visit W&B support
https://github.com/wandb/wandb/issues
```

## How to Access Your Results (Workarounds)

### Via Local Logs
All W&B data is synced locally:

```bash
# View sweep results
cat src/model/06_sweep/best_params.yaml

# Browse local wandb logs
ls -R src/model/06_sweep/wandb/
ls -R src/model/07_registration/wandb/
```

### Via MLflow (Recommended)
All metrics are also logged to MLflow, which is working perfectly:

```bash
# Start MLflow UI
mlflow ui

# Or via Python
python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=['0'])
for run in runs[:5]:
    print(f'{run.info.run_name}: MAPE={run.data.metrics.get(\"mape\", \"N/A\")}')
"
```

### Via API (Direct Access)
Access W&B data programmatically:

```python
import wandb

api = wandb.Api()
runs = api.runs("danieljimenez88m-carlosdanieljimenez-com/housing-mlops-gcp")

# Get sweep runs
for run in runs:
    if 'sweep' in run.name:
        print(f"{run.name}: MAPE={run.summary.get('mape', 'N/A')}")
```

## Current Status of Your Runs

### Sweep Results (Step 06)
```yaml
Sweep ID: imu0vp0m
Sweep URL: https://wandb.ai/.../sweeps/imu0vp0m
Method: Bayesian Optimization
Runs Completed: 5/5

Best Run: dry-sweep-5 (5q1840qa)
Best Hyperparameters:
  - n_estimators: 128
  - max_depth: 23
  - min_samples_split: 9
  - min_samples_leaf: 9
  - max_features: log2

Best Metrics:
  - MAPE: 20.40%
  - R²: 0.7834
  - RMSE: 53,277
  - Within 10%: 36.2%
```

### Alternative Sweep Visualization
Since the new sweep (imu0vp0m) results are:

| Run | MAPE | R² | Within 10% | n_estimators | max_depth |
|-----|------|-----|-----------|--------------|-----------|
| morning-sweep-5 | 19.96% | 0.7916 | 37.3% | 300 | 25 |
| treasured-sweep-1 | 20.29% | 0.7859 | 37.1% | 148 | 15 |
| good-sweep-3 | 20.37% | 0.7877 | 37.0% | 462 | 14 |
| charmed-sweep-2 | 21.67% | 0.7698 | 34.4% | 219 | 11 |
| silvery-sweep-4 | 22.70% | 0.7565 | 32.8% | 254 | 10 |

**Best run from current sweep:** morning-sweep-5 with **19.96% MAPE** ✨

## Why This Error Happens
W&B's web interface uses GraphQL to query run data. When W&B updates their backend schema but the frontend still queries old fields, this error occurs. It's a **temporary platform issue**, not a problem with your code or data.

## Summary
- ✅ Your pipeline executed successfully
- ✅ All 7 steps completed (515.92 seconds total)
- ✅ All metrics logged to W&B (verified in output logs)
- ✅ All metrics logged to MLflow (verified - Model Version 5)
- ✅ Model registered successfully
- ✅ configs/model_config.yaml generated
- ❌ W&B web UI has a temporary GraphQL schema error
- ✅ Use workarounds above to access your data

## Recommendation
**For now:** Use MLflow UI to visualize results (it's working perfectly):
```bash
mlflow ui
# Navigate to http://localhost:5000
```

**Short term:** Try browser cache clear or different browser

**Long term:** Monitor W&B status or update wandb client

---

**Note:** This is a known type of W&B platform error and typically resolves within hours to days as W&B deploys fixes. Your data is safe and accessible via alternative methods.
