"""Test-only entry for the MRI <-> CT baselines (CT -> MRI and MRI -> CT on the test split).

Takes the same arguments as train_mri_ct.py, with the checkpoint given by --checkpoint_path,
and runs its test-only mode.
"""
import sys

from train_mri_ct import main

if __name__ == "__main__":
    main(sys.argv[1:] + ["--test"])
