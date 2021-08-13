# mind cluster
module load anaconda3
srun -p gpu --cpus-per-task=4 --gres=gpu:1 --mem=40GB --time=24:00:00 --nodelist=mind-1-23 --pty $SHELL
module load cuda-11.1.1 cudnn-11.1.1-v8.0.4.30
export LD_LIBRARY_PATH=/opt/cudnn/cuda-11.1/8.0.4.30/cuda/lib64:/opt/cuda/11.1.1/lib64:/opt/cuda/11.1.1/extras/CUPTI/lib64
cd infopt
poetry shell

# -----------------------------------------------------------------------------
# 1. Ackley 5D with TF2 (NN-INF & NN-MCD)
# -----------------------------------------------------------------------------

for i in $(seq -w 1 20); do
    # NN-INF (TF2)
    python exp_sim_optim.py run \
        --fname ackley --fdim 5 --init-points 1 --optim-iters 500 \
        --save-file "results/ackley5d/nn_tf2_$i.pkl" --tb-dir "logdir/ackley5d/nn_tf2_$i" \
        nn_tf2 --layer-sizes "16,16,8" --bom-optim-params "learning_rate=0.02" --bom-weight-decay 1e-4
done

for i in $(seq -w 1 20); do
    # NN-MCD (TF2)
    python exp_sim_optim.py run \
        --fname ackley --fdim 5 --init-points 1 --optim-iters 500 \
        --save-file "results/ackley5d/nnmcd_tf2_$i.pkl" --tb-dir "logdir/ackley5d/nnmcd_tf2_$i" \
        nnmcd_tf2 --layer-sizes "16,16,8" \
        --bom-optim-params "learning_rate=0.02" --mcd-dropout 0.25 --mcd-lengthscale 1e-2 --mcd-tau 0.25
done

for i in $(seq -w 1 20); do
    # NN-INF (torch)
    python exp_sim_optim.py run \
        --fname ackley --fdim 5 --init-points 1 --optim-iters 500 \
        --save-file "results/ackley5d/nn_wd_$i.pkl" --tb-dir "logdir/ackley5d/nn_wd_$i" \
        nn --layer-sizes "16,16,8" --bom-optim-params "lr=0.02,weight_decay=1e-4"
done

#
# Plotting
#

python exp_sim_optim.py plot-regrets --res-dir results/ackley5d \
    --save-file results/ackley5d/regrets_tf2.pdf \
    --skip-pats "results/ackley5d/nn_lbfgs[0-9]+.pkl"

python exp_sim_optim.py plot-timing --res-dir results/ackley5d \
    --save-file results/ackley5d/timing_tf2.pdf \
    --skip-pats "results/ackley5d/nn_lbfgs[0-9]+.pkl"
