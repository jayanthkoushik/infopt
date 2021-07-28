"""run_exp_sim_optim_simple.py:
    runs multiple sessions of exp_sim_optim_simple.py."""

import click
import os.path

from exp_sim_optim_simple import(
    setup_objective,
    run_sim_optim_mcd_tf2,
    run_sim_optim_tf2,
    run_sim_optim_torch,
    run_sim_optim_gp,
)


@click.command()
@click.argument("method",
                type=click.Choice(["all", "gplcb", "nnmcd", "nninf", "nninf_torch"]))
@click.option("--objective", default="ackley",
              type=click.Choice(["ackley", "rastrigin", "randomnn"]),
              help="objective function")
@click.option("--input-dim", default=10, help="input space dimension")
@click.option("--init-points", default=10, help="initial # of training points")
@click.option("--max-iter", default=500, help="maximum # of acq iterations")
@click.option("--n-runs", default=20, help="# of repeated runs")
@click.option("--layer-sizes", "-l", type=int, multiple=True,
              default=[32, 32, 32, 32, 32, 32, 16],
              help="number of units per layer for NN models")
@click.option("--save-dir", default="simple_results_v2",
              help="directory to save results")
def run_exp_sim_optim_simple(method, objective, input_dim, init_points,
                             max_iter, n_runs, layer_sizes, save_dir):
    """runs multiple sessions of exp_sim_optim_simple.py.

    Tested configurations:
        objective="ackley", input_dim=2, init_points=1, max_iter=100, n_runs=20, layer_sizes=[16, 16, 8],
        nn_optim_cls={"lr": 0.02, "weight_decay": 0.0}, dropout=0.25 if method=="nnmcd" else 0.1

        objective="ackley", input_dim=10, init_points=1, max_iter=300, n_runs=20, layer_sizes=[100, 100, 100],
        nn_optim_cls={"lr": 0.002, "weight_decay": 1e-4}, dropout=0.25

        (v2)
        objective="ackley", input_dim=10, init_points=10, max_iter=200, n_runs=20,
        layer_sizes=[32, 32, 32, 32, 16],
        nn_optim_cls={"lr": 0.005, "weight_decay": 1e-4}, dropout=0.25 if method=="nnmcd" else 0.1
        objective="ackley", input_dim=10, init_points=10, max_iter=500, n_runs=20,
        layer_sizes=[32, 32, 32, 32, 32, 32, 16],
        nn_optim_cls={"lr": 0.005, "weight_decay": 1e-4}, dropout=0.25 if method=="nnmcd" else 0.1
    """

    save_dir = os.path.join(save_dir, f"{objective.lower()}{input_dim}d")
    os.makedirs(save_dir, exist_ok=True)
    layer_sizes = [int(sz) for sz in layer_sizes]

    for i in range(n_runs):
        print(f"================ Run {i:02d} ================")

        # Setup
        problem, domain = setup_objective(
            objective,
            input_dim,
            init_points,
        )

        if method == "all" or method == "nnmcd":
            print("******** NN-MCD ********")
            run_sim_optim_mcd_tf2(
                problem,
                domain,
                layer_sizes=layer_sizes,
                dropout=0.1,
                n_dropout_samples=100,
                lengthscale=1e-2,
                tau=0.25,
                max_iter=max_iter,
                init_points=init_points,
                save_filename=os.path.join(save_dir, f"nnmcd{i:02d}.pkl"),
            )

        if method == "all" or method == "nninf":
            print("******** NN-INF ********")
            run_sim_optim_tf2(
                problem,
                domain,
                layer_sizes=layer_sizes,
                ihvp_rank=20,
                ihvp_batch_size=16,
                nn_optim_params={"learning_rate": 0.005},
                weight_decay=1e-4,
                dropout=0.1,
                max_iter=max_iter,
                num_higs=16,
                ihvp_n=30,
                init_points=init_points,
                save_filename=os.path.join(save_dir, f"nninf{i:02d}.pkl"),
            )

        if method == "all" or method == "nninf_torch":
            print("******** NN-INF (torch) ********")
            run_sim_optim_torch(
                problem,
                domain,
                layer_sizes=layer_sizes,
                ihvp_rank=20,
                ihvp_batch_size=16,
                nn_optim_params={"lr": 0.005, "weight_decay": 1e-4},
                dropout=0.1,
                max_iter=max_iter,
                num_higs=16,
                ihvp_n=30,
                init_points=init_points,
                save_filename=os.path.join(save_dir, f"nninf_torch{i:02d}.pkl"),
            )

        if method == "all" or method == "gplcb":
            print("******** GP-LCB ********")
            run_sim_optim_gp(
                problem,
                domain,
                max_iter=max_iter,
                acquisition_type="LCB",
                init_points=init_points,
                save_filename=os.path.join(save_dir, f"gplcb{i:02d}.pkl"),
                report_filename=os.path.join(save_dir, f"gplcb{i:02d}.report"),
                evaluations_filename=os.path.join(save_dir, f"gplcb{i:02d}.evals"),
            )


if __name__ == '__main__':
    run_exp_sim_optim_simple()
