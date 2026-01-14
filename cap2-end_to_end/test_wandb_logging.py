"""
Test script to verify W&B logging works correctly
"""
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set W&B environment variables
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_SILENT"] = "false"
os.environ["WANDB_CONSOLE"] = "wrap"

print("Testing W&B logging...")
print(f"WANDB_API_KEY set: {bool(os.getenv('WANDB_API_KEY'))}")
print(f"WANDB_PROJECT: {os.getenv('WANDB_PROJECT', 'housing-mlops-gcp')}")

# Initialize W&B with explicit settings
wandb_settings = wandb.Settings(
    console="wrap"
)

run = wandb.init(
    project=os.getenv('WANDB_PROJECT', 'housing-mlops-gcp'),
    name="test_wandb_logging",
    job_type="test",
    settings=wandb_settings
)

print(f"W&B Run ID: {run.id}")
print(f"W&B Run URL: {run.url}")

# Test 1: Log metrics
print("\nTest 1: Logging metrics...")
wandb.log({"test_metric": 42, "test_accuracy": 0.95})
print("✓ Metrics logged")

# Test 2: Log system metrics
print("\nTest 2: Logging system info...")
wandb.log({"cpu_count": os.cpu_count()})
print("✓ System info logged")

# Test 3: Create and log image
print("\nTest 3: Creating and logging image...")
artifacts_dir = Path("artifacts/test")
artifacts_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Test Plot")
plt.xlabel("X")
plt.ylabel("Sin(X)")

plot_path = artifacts_dir / "test_plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Plot saved to: {plot_path}")
wandb.log({"test_plot": wandb.Image(str(plot_path))})
print("✓ Image logged")

# Test 4: Log table
print("\nTest 4: Logging table...")
test_table = wandb.Table(
    columns=["name", "value"],
    data=[["metric1", 0.9], ["metric2", 0.8]]
)
wandb.log({"test_table": test_table})
print("✓ Table logged")

# Finish run with explicit wait
print("\nFinishing W&B run...")
run.finish()

print("\n✅ All tests passed!")
print(f"Check your run at: {run.url}")
