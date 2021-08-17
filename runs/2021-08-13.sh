# mind cluster
module load anaconda3
srun -p gpu --cpus-per-task=4 --gres=gpu:1 --mem=32GB --time=24:00:00 --nodelist=mind-1-26 --pty $SHELL
module load cuda-11.1.1 cudnn-11.1.1-v8.0.4.30
export LD_LIBRARY_PATH=/opt/cudnn/cuda-11.1/8.0.4.30/cuda/lib64:/opt/cuda/11.1.1/lib64:/opt/cuda/11.1.1/extras/CUPTI/lib64
cd infopt
poetry shell


# NNINF Diagnostic (f*=-)
python exp_bpnn_optim.py run --seed 199 --diagnostic \
    --save-file "results/bpnn_ZrO2/nninf_test.pkl" \
    --tb-dir "logdir/bpnn_ZrO2/nninf_test" \
    --init-points 1000 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
    nninf --layer-sizes 32,32,16 --activation gelu \
    --ihvp-rank 32 --ihvp-batch-size 32 --bom-ihvp-n 32 --bom-n-higs 64 \
    --bom-optim-params "learning_rate=0.002" --bom-weight-decay 1e-5 \
    --bom-up-batch-size 32 --bom-up-iters-per-point 500


# Greedy Diagnostic (f*^=-0.903 after 1; f*=-0.921)
python exp_bpnn_optim.py run --seed 200 --diagnostic --exp-multiplier 0.0 \
    --save-file "results/bpnn_ZrO2/nngreedy_test.pkl" \
    --tb-dir "logdir/bpnn_ZrO2/nngreedy_test" \
    --init-points 300 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
    nninf --layer-sizes 32,32 --activation gelu \
    --bom-optim-params "learning_rate=0.002" --bom-weight-decay 1e-5 \
    --bom-up-batch-size 32 --bom-up-iters-per-point 500


# -----------------------------------------------------------------------------
# 1. BPNN on Synthetic Data
# -----------------------------------------------------------------------------

# Full runs
for i in $(seq -w 1 10); do
    # NN-MCD
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/nnmcd_$i.pkl" \
        --tb-dir "logdir/bpnn_ZrO2/nnmcd_$i" \
        --init-points 300 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
        nnmcd --layer-sizes 32,32 --activation gelu \
        --bom-optim-params "learning_rate=0.002" \
        --bom-up-batch-size 32 --bom-up-iters-per-point 500 \
        --mcd-dropout 0.1 --mcd-lengthscale 1.0 --mcd-tau 1.0
    # NN-Greedy
    python exp_bpnn_optim.py run --seed $i --exp-multiplier 0.0 \
        --save-file "results/bpnn_ZrO2/nngreedy_$i.pkl" \
        --tb-dir "logdir/bpnn_ZrO2/nngreedy_$i" \
        --init-points 300 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
        nninf --layer-sizes 32,32 --activation gelu \
        --bom-optim-params "learning_rate=0.002" --bom-weight-decay 1e-5 \
        --bom-up-batch-size 32 --bom-up-iters-per-point 500
    # NN-INF
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/nninf_$i.pkl" \
        --tb-dir "logdir/bpnn_ZrO2/nninf_$i" \
        --init-points 300 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
        nninf --layer-sizes 32,32 --activation gelu \
        --bom-optim-params "learning_rate=0.002" --bom-weight-decay 1e-5 \
        --bom-up-batch-size 32 --bom-up-iters-per-point 500
done

# Random
for i in $(seq -w 1 20); do
    python exp_bpnn_optim.py run --seed $i \
          --save-file "results/bpnn_ZrO2/random_$i.pkl" \
          --init-points 300 --optim-iters 100 \
          random
done

python exp_bpnn_optim.py plot-regrets --res-dir results/bpnn_ZrO2 \
    --save-file results/bpnn_ZrO2/regrets.pdf --skip-pats "results/bpnn_ZrO2/old"
python exp_bpnn_optim.py plot-timing --res-dir results/bpnn_ZrO2 \
    --save-file results/bpnn_ZrO2/timing.pdf --skip-pats "results/bpnn_ZrO2/old"
python exp_bpnn_optim.py plot-mu --res-dir results/bpnn_ZrO2 \
    --save-file results/bpnn_ZrO2/mu.pdf --skip-pats "results/bpnn_ZrO2/old"
python exp_bpnn_optim.py plot-sigma --res-dir results/bpnn_ZrO2 \
    --save-file results/bpnn_ZrO2/sigma.pdf --skip-pats "results/bpnn_ZrO2/old" "results/bpnn_ZrO2/nngreedy"
python exp_bpnn_optim.py plot-acq --res-dir results/bpnn_ZrO2 \
    --save-file results/bpnn_ZrO2/acq.pdf --skip-pats "results/bpnn_ZrO2/old" "results/bpnn_ZrO2/nngreedy"

# --diagnostic
#python exp_bpnn_optim.py plot-mse --res-dir results/bpnn_ZrO2 \
#    --save-file results/bpnn_ZrO2/mse.pdf --skip-pats "results/bpnn_ZrO2/old"
#python exp_bpnn_optim.py plot-mse_test --res-dir results/bpnn_ZrO2 \
#    --save-file results/bpnn_ZrO2/mse_test.pdf --skip-pats "results/bpnn_ZrO2/old"

# -----------------------------------------------------------------------------
# 2. BPNN on Synthetic Data, Re-run
#
# - Make converge at each step
# - Use a deeper network (32, 32, 16)
# - 1000 initial points, 100 candidates, no upsampling, 4 interval
# - Increase IHVP samples for larger network
# - Increase MC dropout and lower lengthscale (uncertainty saturates at 1.00)
# -----------------------------------------------------------------------------

for i in $(seq -w 201 210); do
    python exp_bpnn_optim.py run --seed $i \
          --save-file "results/bpnn_ZrO2_v2/random_$i.pkl" \
          --init-points 1000 --optim-iters 100 \
          random
done

# Full runs
for i in $(seq -w 201 210); do
    # NN-INF
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2_v2/nninf_$i.pkl" \
        --tb-dir "logdir/bpnn_ZrO2_v2/nninf_$i" \
        --init-points 1000 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
        nninf --layer-sizes 32,32,16 --activation gelu \
        --ihvp-rank 32 --ihvp-batch-size 32 --bom-ihvp-n 32 --bom-n-higs 64 \
        --bom-optim-params "learning_rate=0.002" --bom-weight-decay 1e-5 \
        --bom-up-batch-size 32 --bom-up-iters-per-point 500
    # NN-Greedy
    python exp_bpnn_optim.py run --seed $i --exp-multiplier 0.0 \
        --save-file "results/bpnn_ZrO2_v2/nngreedy_$i.pkl" \
        --tb-dir "logdir/bpnn_ZrO2_v2/nngreedy_$i" \
        --init-points 1000 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
        nninf --layer-sizes 32,32,16 --activation gelu \
        --ihvp-rank 1 --bom-ihvp-n 1 --bom-n-higs 1 \
        --bom-optim-params "learning_rate=0.002" --bom-weight-decay 1e-5 \
        --bom-up-batch-size 32 --bom-up-iters-per-point 500
    # NN-MCD
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2_v2/nnmcd_$i.pkl" \
        --tb-dir "logdir/bpnn_ZrO2_v2/nnmcd_$i" \
        --init-points 1000 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
        nnmcd --layer-sizes 32,32,16 --activation gelu \
        --bom-optim-params "learning_rate=0.002" \
        --bom-up-batch-size 32 --bom-up-iters-per-point 500 \
        --mcd-dropout 0.25 --mcd-lengthscale 0.1 --mcd-tau 1.0
done

python exp_bpnn_optim.py plot-regrets --res-dir results/bpnn_ZrO2_v2 \
    --save-file results/bpnn_ZrO2_v2/regrets.pdf
python exp_bpnn_optim.py plot-timing --res-dir results/bpnn_ZrO2_v2 \
    --save-file results/bpnn_ZrO2_v2/timing.pdf
python exp_bpnn_optim.py plot-sigma --res-dir results/bpnn_ZrO2_v2 \
    --save-file results/bpnn_ZrO2_v2/sigma.pdf --skip-pats "results/bpnn_ZrO2_v2/nngreedy"


python exp_bpnn_optim.py evaluate-retrieval \
        --res-pats "results/bpnn_ZrO2_v2/nngreedy*.pkl" \
        --save-file "results/bpnn_ZrO2_v2/nngreedy_retrieval.csv"
python exp_bpnn_optim.py evaluate-retrieval \
        --res-pats "results/bpnn_ZrO2_v2/nninf*.pkl" \
        --save-file "results/bpnn_ZrO2_v2/nninf_retrieval.csv"
python exp_bpnn_optim.py evaluate-retrieval \
        --res-pats "results/bpnn_ZrO2_v2/nnmcd*.pkl" \
        --save-file "results/bpnn_ZrO2_v2/nnmcd_retrieval.csv"
python exp_bpnn_optim.py evaluate-retrieval \
        --res-pats "results/bpnn_ZrO2_v2/random*.pkl" \
        --save-file "results/bpnn_ZrO2_v2/random_retrieval.csv"

# -----------------------------------------------------------------------------
# 3. BPNN with less pre-training
#
# - Make converge at each step
# - Use a deeper network (32, 32, 16)
# - 100 initial points, 100 candidates, no upsampling, 4 interval
# - Increase IHVP samples for larger network
# - Increase MC dropout and lower lengthscale (uncertainty saturates at 1.00)
# -----------------------------------------------------------------------------

# NNINF Diagnostic (f*=-)
python exp_bpnn_optim.py run --seed 198 --diagnostic \
    --save-file "results/bpnn_ZrO2_v3/nninf_test.pkl" \
    --tb-dir "logdir/bpnn_ZrO2_v3/nninf_test" \
    --init-points 100 --optim-iters 100 --model-update-interval 4 --acq-n-candidates 100 \
    nninf --layer-sizes 32,32,16 --activation gelu \
    --bom-optim-params "learning_rate=0.002" --bom-weight-decay 1e-5 \
    --bom-up-batch-size 32 --bom-up-iters-per-point 500

#     --ihvp-rank 32 --ihvp-batch-size 32 --bom-ihvp-n 32 --bom-n-higs 64 \