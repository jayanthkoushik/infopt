# mind cluster
module load anaconda3
srun -p gpu --cpus-per-task=4 --gres=gpu:1 --mem=40GB --time=24:00:00 --nodelist=mind-1-30 --pty $SHELL
module load cuda-11.1.1 cudnn-11.1.1-v8.0.4.30
export LD_LIBRARY_PATH=/opt/cudnn/cuda-11.1/8.0.4.30/cuda/lib64:/opt/cuda/11.1.1/lib64:/opt/cuda/11.1.1/extras/CUPTI/lib64
cd infopt
poetry shell

# -----------------------------------------------------------------------------
# 1. BPNN on Synthetic Data
# -----------------------------------------------------------------------------

for i in $(seq -w 1 20); do
    # NN-Greedy
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/nngreedy_$i.pkl" --tb-dir "logdir/bpnn_ZrO2/nngreedy_$i" \
        --n-search 1200 --n-test 500 --exp-multiplier 0.0 \
        --init-points 200 --optim-iters 200 --model-update-interval 8 \
        nninf --layer-sizes 64,64 --activation relu \
        --bom-optim-params "learning_rate=0.005" --bom-weight-decay 1e-4 \
        --bom-up-iters-per-point 50 --bom-up-upsample-new 0.25 \
        --pretrain-epochs 50
    # NN-MCD
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/nnmcd_$i.pkl" --tb-dir "logdir/bpnn_ZrO2/nnmcd_$i" \
        --n-search 1200 --n-test 500 \
        --init-points 200 --optim-iters 200 --model-update-interval 8 \
        nnmcd --layer-sizes 64,64 --activation relu \
        --bom-optim-params "learning_rate=0.005" \
        --bom-up-iters-per-point 50 --bom-up-upsample-new 0.25 \
        --mcd-dropout 0.1 --mcd-lengthscale 1e-2 --mcd-tau 0.1 \
        --pretrain-epochs 50
    # NN-INF
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/nninf_$i.pkl" --tb-dir "logdir/bpnn_ZrO2/nninf_$i" \
        --n-search 1200 --n-test 500 \
        --init-points 200 --optim-iters 200 --model-update-interval 8 \
        nninf --layer-sizes 64,64 --activation relu \
        --bom-optim-params "learning_rate=0.005" --bom-weight-decay 1e-4 \
        --bom-up-iters-per-point 50 --bom-up-upsample-new 0.25 \
        --pretrain-epochs 50
done
