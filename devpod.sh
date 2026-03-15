#!/usr/bin/env bash
set -euo pipefail

SSH_CONFIG="$HOME/.ssh/config"
REPO_URL="git@github.com:cybershiptrooper/forking-paths-clone.git"
REPO_DIR="forking-paths-clone"
BRANCH="cot_rewards"

usage() {
    echo "Usage: $0 [--start | --stop]"
    echo "  --start   Launch a dev pod, update SSH config, and set up the repo"
    echo "  --stop    Stop the running dev pod"
    # exit 1
}

update_ssh_config() {
    local new_host="$1"
    local tmp_config
    tmp_config=$(mktemp)

    # Check if pi-mentee-gpu block exists (commented or not)
    if grep -q "pi-mentee-gpu" "$SSH_CONFIG"; then
        # Remove existing pi-mentee-gpu block (commented or uncommented)
        awk '
        /^#?\s*Host pi-mentee-gpu/ { skip=1; next }
        skip && /^#?\s*(Hostname|User|ProxyJump|ForwardAgent|AddKeysToAgent|StrictHostKeyChecking|UserKnownHostsFile)\s/ { next }
        skip && /^$/ { skip=0; next }
        skip && /^#?\s*Host\s/ { skip=0 }
        !skip { print }
        ' "$SSH_CONFIG" > "$tmp_config"
    else
        cp "$SSH_CONFIG" "$tmp_config"
    fi

    # Append new pi-mentee-gpu block
    cat >> "$tmp_config" <<EOF

Host pi-mentee-gpu
    Hostname ${new_host}
    User rgupta
    ProxyJump pi-mentee-login
    ForwardAgent yes
    AddKeysToAgent yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF

    cp "$tmp_config" "$SSH_CONFIG"
    rm "$tmp_config"
    echo "Updated pi-mentee-gpu hostname to: ${new_host}"
}

start_pod() {
    echo "Starting dev pod on pi-mentee-login..."

    # Start pod and get the node it landed on
    # Use login shell (-l) so that profile/bashrc paths are sourced
    ssh pi-mentee-login "bash -lc 'pod'" || true

    echo "Checking where the pod landed..."
    local node
    node=$(ssh pi-mentee-login "bash -lc 'squeue --me --noheader -o \"%N\"'" 2>/dev/null | awk 'NR==1{print}' | tr -d '[:space:]')

    if [[ -z "$node" ]]; then
        echo "ERROR: No running pod found. Check manually with: ssh pi-mentee-login squeue --me"
        exit 1
    fi

    echo "Pod landed on: ${node}"

    # Update SSH config
    update_ssh_config "$node"

    # Wait a moment for the pod to be ready
    echo "Waiting for pod to be ready..."
    sleep 5

    # Setup repo on the pod
    echo "Setting up repository on the pod..."
    ssh pi-mentee-gpu bash -s <<'REMOTE_SETUP'
set -euo pipefail

REPO_DIR="forking-paths-clone"
REPO_URL="git@github.com:cybershiptrooper/forking-paths-clone.git"
BRANCH="cot_rewards"

if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$REPO_DIR"
    git fetch --all
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
    git pull origin "$BRANCH" || true
else
    echo "Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
fi

echo "Setting up Python environment with uv..."
uv sync

echo "Done! Repository is ready at ~/$REPO_DIR"
REMOTE_SETUP

    echo ""
    echo "Dev pod is ready! Connect with: ssh pi-mentee-gpu"
}

stop_pod() {
    echo "Current jobs:"
    ssh pi-mentee-login "bash -lc 'squeue --me'" || true

    echo ""
    echo "Cancelling all jobs..."
    ssh pi-mentee-login "bash -lc 'scancel --me && echo Cancelled successfully || echo Failed to cancel'"

    echo ""
    echo "Remaining jobs:"
    ssh pi-mentee-login "bash -lc 'squeue --me'" || true
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    usage
fi

case "$1" in
    --start)
        start_pod
        ;;
    --stop)
        stop_pod
        ;;
    *)
        usage
        ;;
esac
