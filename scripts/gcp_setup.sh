#!/bin/bash
# GCP GPU VM Setup for Parameter Golf Experiments
#
# Creates a VM with a GPU in us-central1, installs dependencies,
# and downloads the FineWeb dataset.
#
# Usage:
#   bash scripts/gcp_setup.sh [--gpu-type nvidia-tesla-t4|nvidia-tesla-a100|nvidia-h100-80gb]
#   bash scripts/gcp_setup.sh --dry-run  # Print commands without executing
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Project: testingout-423013
#   - Compute Engine API enabled
#   - GPU quota available

set -euo pipefail

# Configuration
PROJECT="testingout-423013"
ZONE="us-central1-a"
VM_NAME="param-golf-gpu"
GPU_TYPE="${1:-nvidia-tesla-a100}"
DRY_RUN=false

# Parse args
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --gpu-type=*) GPU_TYPE="${arg#*=}" ;;
        nvidia-*) GPU_TYPE="$arg" ;;
    esac
done

# Machine type mapping
case $GPU_TYPE in
    nvidia-tesla-t4)
        MACHINE_TYPE="n1-standard-8"
        GPU_COUNT=1
        COST_HR="~\$0.95/hr"
        ;;
    nvidia-tesla-a100|nvidia-a100-80gb)
        MACHINE_TYPE="a2-highgpu-1g"
        GPU_COUNT=1
        COST_HR="~\$3.67/hr"
        ;;
    nvidia-h100-80gb)
        MACHINE_TYPE="a3-highgpu-1g"
        GPU_COUNT=1
        COST_HR="~\$5.50/hr"
        ;;
    *)
        echo "Unknown GPU type: $GPU_TYPE"
        echo "Choose from: nvidia-tesla-t4, nvidia-tesla-a100, nvidia-h100-80gb"
        exit 1
        ;;
esac

echo "========================================"
echo "  Parameter Golf GCP GPU Setup"
echo "========================================"
echo "  Project:  $PROJECT"
echo "  Zone:     $ZONE"
echo "  VM:       $VM_NAME"
echo "  GPU:      $GPU_TYPE ($GPU_COUNT)"
echo "  Machine:  $MACHINE_TYPE"
echo "  Cost:     $COST_HR"
echo "========================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would execute the following commands:"
    echo ""
fi

run_cmd() {
    echo "$ $*"
    if ! $DRY_RUN; then
        eval "$@"
    fi
}

# Create VM
echo ">>> Creating VM..."
run_cmd "gcloud compute instances create $VM_NAME \\
    --project=$PROJECT \\
    --zone=$ZONE \\
    --machine-type=$MACHINE_TYPE \\
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \\
    --boot-disk-size=200GB \\
    --boot-disk-type=pd-ssd \\
    --image-family=pytorch-latest-gpu \\
    --image-project=deeplearning-platform-release \\
    --maintenance-policy=TERMINATE \\
    --metadata=install-nvidia-driver=True \\
    --scopes=default,storage-rw"

echo ""
echo ">>> Waiting for VM to be ready..."
if ! $DRY_RUN; then
    sleep 30
fi

# Setup script to run on the VM
SETUP_SCRIPT='#!/bin/bash
set -e
echo "=== Setting up Parameter Golf environment ==="

# Install system deps
sudo apt-get update -qq && sudo apt-get install -y -qq zstd

# Create workspace
mkdir -p ~/param-golf && cd ~/param-golf

# Clone or setup
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi

# Install Python deps
pip install -q numpy tqdm torch huggingface-hub setuptools "typing-extensions==4.15.0" \
    datasets tiktoken sentencepiece zstandard

# Download full dataset (80 shards)
cd parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

echo "=== Setup complete! ==="
echo "Dataset: ~/param-golf/parameter-golf/data/datasets/fineweb10B_sp1024/"
echo "To run training:"
echo "  cd ~/param-golf/parameter-golf && python train_gpt.py"
'

echo ">>> Running setup on VM..."
if ! $DRY_RUN; then
    echo "$SETUP_SCRIPT" | gcloud compute ssh $VM_NAME \
        --project=$PROJECT --zone=$ZONE -- 'bash -s'
fi

echo ""
echo "========================================"
echo "  VM is ready!"
echo "========================================"
echo ""
echo "SSH into the VM:"
echo "  gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE"
echo ""
echo "Copy experiments to VM:"
echo "  bash scripts/run_on_gcp.sh sync"
echo ""
echo "IMPORTANT: Stop the VM when done to save credits:"
echo "  gcloud compute instances stop $VM_NAME --project=$PROJECT --zone=$ZONE"
echo ""
echo "Delete VM when finished:"
echo "  gcloud compute instances delete $VM_NAME --project=$PROJECT --zone=$ZONE"
