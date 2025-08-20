
from pdb import run


if __name__ == "__main__":
    from sae_insights import run_experiment_loop, run_non_private_baseline
    from dense_representation import run_experiment_loop as run_dense_experiment_loop, run_non_private_baseline as run_dense_non_private_baseline
    run_experiment_loop()
    run_non_private_baseline()
    run_dense_experiment_loop()
    run_dense_non_private_baseline()
