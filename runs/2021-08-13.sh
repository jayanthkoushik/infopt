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
    # NN-INF
    python exp_bpnn_optim.py run --seed $i --acq-n-candidates 50 \
        --optim-iters 200 --model-update-interval 4 --exp-multiplier 1.0 \
        --save-file "results/bpnn_synthetic/nninf_$i.pkl" \
        --tb-dir "logdir/bpnn_synthetic/nninf_$i" \
        nninf --layer-sizes 32,32 --activation tanh \
        --bom-optim-params "learning_rate=0.01" --dropout 0.1 \
        --pretrain-epochs 50
    # NN-Greedy
    python exp_bpnn_optim.py run --seed $i --acq-n-candidates 50 \
        --optim-iters 200 --model-update-interval 4 --exp-multiplier 0.0 \
        --save-file "results/bpnn_synthetic/nngreedy_$i.pkl" \
        --tb-dir "logdir/bpnn_synthetic/nngreedy_$i" \
        nninf --layer-sizes 32,32 --activation tanh \
        --bom-optim-params "learning_rate=0.01" --dropout 0.1 \
        --pretrain-epochs 50
    # NN-MCD
    python exp_bpnn_optim.py run --seed $i --acq-n-candidates 50 \
        --optim-iters 200 --model-update-interval 4 --exp-multiplier 1.0 \
        --save-file "results/bpnn_synthetic/nnmcd_$i.pkl" \
        --tb-dir "logdir/bpnn_synthetic/nnmcd_$i" \
        nnmcd --layer-sizes 32,32 --activation tanh \
        --bom-optim-params "learning_rate=0.01" \
        --mcd-dropout 0.25 --mcd-lengthscale 1e-2 --mcd-tau 0.25 \
        --pretrain-epochs 50
done
