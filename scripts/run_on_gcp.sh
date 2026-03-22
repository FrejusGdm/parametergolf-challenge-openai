#!/bin/bash
# Helper to sync experiments to GCP VM and run them
#
# Usage:
#   bash scripts/run_on_gcp.sh sync       # Copy experiments + scripts to VM
#   bash scripts/run_on_gcp.sh run CMD    # Run a command on the VM
#   bash scripts/run_on_gcp.sh ssh        # Open SSH session
#   bash scripts/run_on_gcp.sh stop       # Stop VM (save credits)
#   bash scripts/run_on_gcp.sh start      # Start VM

set -euo pipefail

PROJECT="testingout-423013"
ZONE="us-central1-a"
VM_NAME="param-golf-gpu"
REMOTE_DIR="~/param-golf"

CMD="${1:-help}"

case $CMD in
    sync)
        echo "Syncing experiments and scripts to VM..."
        gcloud compute scp --recurse --project=$PROJECT --zone=$ZONE \
            experiments scripts notebooks \
            $VM_NAME:$REMOTE_DIR/
        echo "Done! Files synced to $REMOTE_DIR/ on $VM_NAME"
        ;;
    run)
        shift
        echo "Running on VM: $*"
        gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE \
            -- "cd $REMOTE_DIR && $*"
        ;;
    ssh)
        echo "Connecting to $VM_NAME..."
        gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE
        ;;
    stop)
        echo "Stopping $VM_NAME..."
        gcloud compute instances stop $VM_NAME --project=$PROJECT --zone=$ZONE
        echo "VM stopped. Credits are no longer being consumed."
        ;;
    start)
        echo "Starting $VM_NAME..."
        gcloud compute instances start $VM_NAME --project=$PROJECT --zone=$ZONE
        echo "VM starting... wait ~30s for it to be ready."
        ;;
    status)
        gcloud compute instances describe $VM_NAME --project=$PROJECT --zone=$ZONE \
            --format="table(name,status,zone,machineType,scheduling.preemptible)"
        ;;
    results)
        echo "Fetching results from VM..."
        gcloud compute scp --recurse --project=$PROJECT --zone=$ZONE \
            $VM_NAME:$REMOTE_DIR/experiments/ experiments/
        echo "Results synced to local experiments/"
        ;;
    help|*)
        echo "Usage: bash scripts/run_on_gcp.sh COMMAND"
        echo ""
        echo "Commands:"
        echo "  sync     Copy experiments + scripts to VM"
        echo "  run CMD  Run a command on the VM"
        echo "  ssh      Open SSH session to VM"
        echo "  stop     Stop VM (save credits!)"
        echo "  start    Start VM"
        echo "  status   Check VM status"
        echo "  results  Fetch experiment results from VM"
        ;;
esac
