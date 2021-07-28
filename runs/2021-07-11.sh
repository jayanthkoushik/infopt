# mind cluster
module load anaconda3
srun -p gpu --cpus-per-task=4 --gres=gpu:1 --mem=40GB --time=24:00:00 --pty $SHELL
module load cuda-11.1.1 cudnn-11.1.1-v8.0.4.30
export LD_LIBRARY_PATH=/opt/cudnn/cuda-11.1/8.0.4.30/cuda/lib64:/opt/cuda/11.1.1/lib64:/opt/cuda/11.1.1/extras/CUPTI/lib64
cd infopt
poetry shell

# -----------------------------------------------------------------------------
# 1. Ackley 5D (Adam & L-BFGS)
# -----------------------------------------------------------------------------

# NN-INF
for i in $(seq -w 1 20); do
   python exp_sim_optim.py run \
       --fname ackley --fdim 5 --init-points 1 --optim-iters 500 \
       --save-file "results/ackley5d/nn$i.pkl" --tb-dir "logdir/ackley5d/nn$i" \
       nn --layer-sizes "8,8,4"
done

# NN-INF (LBFGS)
for i in $(seq -w 1 20); do
   python exp_sim_optim.py run \
       --fname ackley --fdim 5 --init-points 1 --optim-iters 500 \
       --save-file "results/ackley5d/nn_lbfgs$i.pkl" --tb-dir "logdir/ackley5d/nn_lbfgs$i" \
       nn --layer-sizes "8,8,4" \
       --acq-optim-cls LBFGS --acq-optim-params "lr=1" --acq-optim-lr-decay-gamma 1.0
done

# NN-INF (LBFGS-Greedy)
for i in $(seq -w 1 20); do
   python exp_sim_optim.py run \
       --fname ackley --fdim 5 --init-points 1 --optim-iters 500 --use-const-exp-w 0.0 \
       --save-file "results/ackley5d/nn_lbfgs_greedy$i.pkl" --tb-dir "logdir/ackley5d/nn_lbfgs_greedy$i" \
       nn --layer-sizes "8,8,4" \
       --acq-optim-cls LBFGS --acq-optim-params "lr=1" --acq-optim-lr-decay-gamma 1.0
done

# GP-LCB
for i in $(seq -w 1 20); do
   python exp_sim_optim.py run \
       --fname ackley --fdim 5 --init-points 1 --optim-iters 500 \
       --save-file "results/ackley5d/gp$i.pkl" --tb-dir "logdir/ackley5d/gp$i" \
       gp --acq-type lcb
done

# GP-INF
for i in $(seq -w 1 20); do
   python exp_sim_optim.py run \
       --fname ackley --fdim 5 --init-points 1 --optim-iters 500 \
       --save-file "results/ackley5d/gpinf$i.pkl" --tb-dir "logdir/ackley5d/gpinf$i" \
       gp --acq-type inf
done

# -----------------------------------------------------------------------------
# 2. Target NN (5x64)
# -----------------------------------------------------------------------------

for i in $(seq -w 1 20); do
    # GP-LCB
    python exp_nnet_optim.py run \
        --obj-in-dim 5 --obj-layer-sizes 64,64,64,64,64 --obj-opt-lbfgs \
        --init-points 1 --optim-iters 300 \
        --save-dir "results/nnet_5x64_5d/gp$i" --tb-dir "logdir/nnet_5x64_5d/gp$i" \
        gp --acq-type lcb
    # NN-INF (small)
    python exp_nnet_optim.py run \
        --obj-in-dim 5 --obj-layer-sizes 64,64,64,64,64 --obj-opt-lbfgs \
        --init-points 1 --optim-iters 300 \
        --save-dir "results/nnet_5x64_5d/nn_small$i" --tb-dir "logdir/nnet_5x64_5d/nn_small$i" \
        --load-target "results/nnet_5x64_5d/gp$i/tnet.pt" \
        nn --model-layer-sizes 8,8,4
    # NN-INF (same)
    python exp_nnet_optim.py run \
        --obj-in-dim 5 --obj-layer-sizes 64,64,64,64,64 --obj-opt-lbfgs \
        --init-points 1 --optim-iters 300 \
        --save-dir "results/nnet_5x64_5d/nn_same$i" --tb-dir "logdir/nnet_5x64_5d/nn_same$i" \
        --load-target "results/nnet_5x64_5d/gp$i/tnet.pt" \
        nn --model-layer-sizes 64,64,64,64,64
done

# -----------------------------------------------------------------------------
# 3. Ackley 5D Offline
#
# srun -p gpu --cpus-per-task=8 --gres=gpu:1 --mem=40GB --pty bash run_ackley5d_offline.sh
# -----------------------------------------------------------------------------

for n in $(seq -w 2 4); do
    let N=10**n
    for i in $(seq -w 1 10); do
        # NN-INF (small)
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 100 \
            --save-file "results/ackley5d_offline/nn_small_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/nn_small_$N"_"$i" \
            nn --layer-sizes "8,8,4"
        # NN-INF (large)
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 100 \
            --save-file "results/ackley5d_offline/nn_large_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/nn_large_$N"_"$i" \
            nn --layer-sizes "32,32,16"
        # NN-INF (wide)
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 100 \
            --save-file "results/ackley5d_offline/nn_wide_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/nn_wide_$N"_"$i" \
            nn --layer-sizes "128,128,64" --bom-optim-params "lr=0.01,weight_decay=1e-4"
        # NN-INF (deep)
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 100 \
            --save-file "results/ackley5d_offline/nn_deep_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/nn_deep_$N"_"$i" \
            nn --layer-sizes "64,64,64,64,32" --bom-optim-params "lr=0.01,weight_decay=1e-4"
    done;
    for i in $(seq -w 1 10); do
        # GP-LCB
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 100 \
            --save-file "results/ackley5d_offline/gp_lcb_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/gp_lcb_$N"_"$i" \
            gp --acq-type lcb
        # GP-INF
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 100 \
            --save-file "results/ackley5d_offline/gp_inf_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/gp_inf_$N"_"$i" \
            gp --acq-type inf
    done
done


# -----------------------------------------------------------------------------
# 4. Ackley 5D Offline w/ Hyperparameter Tuning
# -----------------------------------------------------------------------------

for n in $(seq -w 0 4); do
    let N=10**n
    for i in $(seq -w 1 5); do
        # NN-INF (dropout)
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 10 \
            --save-file "results/ackley5d_offline/nn_deep_dropout_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/nn_deep_dropout_$N"_"$i" \
            nn --layer-sizes "64,64,64,64,32" --dropout 0.3
    done;
done

for n in $(seq -w 0 4); do
    let N=10**n
    for i in $(seq -w 1 5); do
        # NN-INF (wd)
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 10 \
            --save-file "results/ackley5d_offline/nn_deep_wd_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/nn_deep_wd_$N"_"$i" \
            nn --layer-sizes "64,64,64,64,32" --bom-optim-params "lr=0.02,weight_decay=1e-4"
    done;
done

for n in $(seq -w 0 4); do
    let N=10**n
    for i in $(seq -w 1 5); do
        # NN-INF (lr0.002)
        python exp_sim_optim.py run \
            --fname ackley --fdim 5 --init-points $N --optim-iters 10 \
            --save-file "results/ackley5d_offline/nn_deep_lr0.002_$N"_"$i.pkl" \
            --tb-dir "logdir/ackley5d_offline/nn_deep_lr0.002_$N"_"$i" \
            nn --layer-sizes "64,64,64,64,32" --bom-optim-params "lr=0.002"
    done;
done

# -----------------------------------------------------------------------------
# 5. Target NN (5d, 5x64) Offline
# -----------------------------------------------------------------------------

for n in $(seq -w 1 3); do
    let N=10**n
    for i in $(seq -w 1 10); do
        # NN-INF (same)
        python exp_nnet_optim.py run \
            --obj-in-dim 5 --obj-layer-sizes 64,64,64,64,64 --obj-opt-lbfgs \
            --init-points $N --optim-iters 500 \
            --save-dir "results/nnet_5x64_5d_offline/nn_same_$N"_"$i" \
            --tb-dir "logdir/nnet_5x64_5d_offline/nn_same_$N"_"$i" \
            nn --model-layer-sizes 64,64,64,64,64 --bom-optim-params "lr=0.01,weight_decay=1e-4"
        # NN-INF (small)
        python exp_nnet_optim.py run \
            --obj-in-dim 5 --obj-layer-sizes 64,64,64,64,64 --obj-opt-lbfgs \
            --init-points $N --optim-iters 500 \
            --save-dir "results/nnet_5x64_5d_offline/nn_small_$N"_"$i" \
            --tb-dir "logdir/nnet_5x64_5d_offline/nn_small_$N"_"$i" \
            --load-target "results/nnet_5x64_5d_offline/nn_same_$N"_"$i/tnet.pt" \
            nn --model-layer-sizes 32,32,16 --bom-optim-params "lr=0.01,weight_decay=1e-4"
        # GP-LCB
        python exp_nnet_optim.py run \
            --obj-in-dim 5 --obj-layer-sizes 64,64,64,64,64 --obj-opt-lbfgs \
            --init-points $N --optim-iters 50 \
            --save-dir "results/nnet_5x64_5d_offline/gp_lcb_$N"_"$i" \
            --tb-dir "logdir/nnet_5x64_5d_offline/gp_lcb_$N"_"$i" \
            --load-target "results/nnet_5x64_5d_offline/nn_same_$N"_"$i/tnet.pt" \
            gp --acq-type lcb
    done;
done

# -----------------------------------------------------------------------------
# 6. Target NN (25d, 7-layers) Offline
# -----------------------------------------------------------------------------


for n in $(seq -w 1 3); do
    let N=10**n
    for i in $(seq -w 1 10); do
        # NN-INF (same)
        python exp_nnet_optim.py run \
            --obj-in-dim 25 --obj-layer-sizes 128,128,128,64,64,32,32 --obj-opt-lbfgs \
            --init-points $N --optim-iters 500 \
            --save-dir "results/nnet_7layers_25d_offline/nn_same_$N"_"$i" \
            --tb-dir "logdir/nnet_7layers_25d_offline/nn_same_$N"_"$i" \
            nn --model-layer-sizes 128,128,128,64,64,32,32 --bom-optim-params "lr=0.01,weight_decay=1e-4"
        # NN-INF (small)
        python exp_nnet_optim.py run \
            --obj-in-dim 25 --obj-layer-sizes 128,128,128,64,64,32,32 --obj-opt-lbfgs \
            --init-points $N --optim-iters 500 \
            --save-dir "results/nnet_7layers_25d_offline/nn_small_$N"_"$i" \
            --tb-dir "logdir/nnet_7layers_25d_offline/nn_small_$N"_"$i" \
            --load-target "results/nnet_7layers_25d_offline/nn_same_$N"_"$i/tnet.pt" \
            nn --model-layer-sizes 64,64,32 --bom-optim-params "lr=0.01,weight_decay=1e-4"
        # GP-LCB
        python exp_nnet_optim.py run \
            --obj-in-dim 25 --obj-layer-sizes 128,128,128,64,64,32,32 --obj-opt-lbfgs \
            --init-points $N --optim-iters 50 \
            --save-dir "results/nnet_7layers_25d_offline/gp_lcb_$N"_"$i" \
            --tb-dir "logdir/nnet_7layers_25d_offline/gp_lcb_$N"_"$i" \
            --load-target "results/nnet_7layers_25d_offline/nn_same_$N"_"$i/tnet.pt" \
            gp --acq-type lcb
    done;
done
