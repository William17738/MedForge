#!/usr/bin/env python3
"""
MedForge - LLM-Powered Document Processing Pipeline

Main entry point for the parallel document processing system.
"""

import os
import sys


def main():
    """Run the full processing pipeline."""
    # Ensure working directory is correct
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    print("MedForge Document Processing System")
    print("=" * 40)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version.split()[0]}")

    try:
        from run_all import main as run_pipeline

        print("\nStarting processing pipeline...")
        run_pipeline()

        print("\nProcessing completed successfully")
        return 0

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 130

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
