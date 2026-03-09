#!/bin/bash
#SBATCH --job-name=physdrive_stmap
#SBATCH --output=physdrive_stmap.out
#SBATCH --error=physdrive_stmap.err
#SBATCH --partition=ihub
#SBATCH -A mobility_arfs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnode106
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=4G
#SBATCH --time=96:00:00

NUM_GPUS=3

echo "========================================"
echo "PhysDrive STMap Generation Pipeline"
echo "Node: $(hostname)"
echo "GPUs: $NUM_GPUS"
echo "Start: $(date)"
echo "========================================"

source /home2/umang.tiwari/rppg_env/bin/activate
cd /home2/umang.tiwari/repos/HSRD/STMap/PhysDrive || exit 1

# =========================================================
# Step 1: Detect face landmarks from RGB videos (3 GPUs)
# =========================================================
echo ""
echo "===== STEP 1: Landmark Detection (3 GPUs parallel) ====="
echo "Start: $(date)"
for i in $(seq 0 $((NUM_GPUS-1))); do
    python Landmark.py --gpu $i --worker_id $i --num_workers $NUM_GPUS &> logs/landmark_gpu${i}.log &
    echo "  Started worker $i on GPU $i (PID $!)"
done
wait
echo "Step 1 finished: $(date)"

# =========================================================
# Step 2: Align faces using landmarks (CPU-only, parallel)
# =========================================================
echo ""
echo "===== STEP 2: Face Alignment (3 workers parallel) ====="
echo "Start: $(date)"
for i in $(seq 0 $((NUM_GPUS-1))); do
    python Align_Face.py --worker_id $i --num_workers $NUM_GPUS &> logs/align_worker${i}.log &
    echo "  Started worker $i (PID $!)"
done
wait
echo "Step 2 finished: $(date)"

# =========================================================
# Step 3: Generate STMaps from aligned faces (CPU-only, parallel)
# =========================================================
echo ""
echo "===== STEP 3: STMap Generation (3 workers parallel) ====="
echo "Start: $(date)"
for i in $(seq 0 $((NUM_GPUS-1))); do
    python STMap.py --worker_id $i --num_workers $NUM_GPUS &> logs/stmap_worker${i}.log &
    echo "  Started worker $i (PID $!)"
done
wait
echo "Step 3 finished: $(date)"

echo ""
echo "========================================"
echo "STMap pipeline finished"
echo "End: $(date)"
echo "========================================"
sleep infinity