#!/usr/bin/env python3
"""
Quick pipeline runner without Hydra
Author: Carlos Daniel Jiménez
Date: 2025-01-13

Simple wrapper to run main.py with default configuration.
"""

import subprocess
import sys

if __name__ == "__main__":
    print("🚀 Starting MLOps Pipeline...\n")

    try:
        # Run the main pipeline
        result = subprocess.run(
            [sys.executable, "main.py"],
            check=True
        )

        print("\n✅ Pipeline completed successfully!")
        sys.exit(result.returncode)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
