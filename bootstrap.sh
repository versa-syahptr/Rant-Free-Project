#!/usr/bin/env bash
# =============================================================================
# IF5251 Bootstrap Script
# Node: CloudLab bare metal, Ubuntu 24.04, NVIDIA P100
# Usage: bash bootstrap.sh
# =============================================================================
set -euo pipefail

LOG_FILE="/disk/if5251/logs/bootstrap_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /disk/if5251/logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "IF5251 Bootstrap — $(date)"
echo "=========================================="

# =============================================================================
# 1. SYSTEM UPDATE
# =============================================================================
echo "[1/9] System update..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential

# =============================================================================
# 2. DISABLE SWAP (runtime only, tidak edit fstab)
# =============================================================================
echo "[2/9] Disabling swap..."
sudo swapoff -a
echo "Swap status: $(free -h | grep Swap)"

# =============================================================================
# 3. NVIDIA DRIVER
# =============================================================================
echo "[3/9] Checking NVIDIA driver..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "Installing nvidia-driver-580..."
  sudo apt install -y nvidia-driver-580
  echo "REBOOT REQUIRED. Re-run bootstrap.sh after reboot."
  exit 0
else
  echo "NVIDIA driver already installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
fi

# =============================================================================
# 4. NVIDIA CONTAINER TOOLKIT
# =============================================================================
echo "[4/9] Checking NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &>/dev/null; then
  echo "Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt update && sudo apt install -y nvidia-container-toolkit
else
  echo "NVIDIA Container Toolkit already installed: $(nvidia-ctk --version | head -1)"
fi

# =============================================================================
# 5. CDI SPEC (must be before K3s install)
# =============================================================================
echo "[5/9] Generating CDI spec..."
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
echo "CDI spec generated: $(sudo head -2 /etc/cdi/nvidia.yaml)"

# =============================================================================
# 6. K3S
# =============================================================================
echo "[6/9] Checking K3s..."

# Create K3s config BEFORE installing — this is the correct way to set
# default-runtime. Writing directly to containerd config.toml is wrong
# because K3s regenerates that file on every restart.
if [ ! -f /etc/rancher/k3s/config.yaml ]; then
  echo "Creating /etc/rancher/k3s/config.yaml..."
  sudo mkdir -p /etc/rancher/k3s
  sudo tee /etc/rancher/k3s/config.yaml << 'EOF'
default-runtime: nvidia
write-kubeconfig-mode: "644"
disable:
  - traefik
EOF
else
  echo "/etc/rancher/k3s/config.yaml already exists"
  # Ensure default-runtime is set even if file pre-exists
  if ! sudo grep -q "default-runtime" /etc/rancher/k3s/config.yaml; then
    echo "Adding default-runtime: nvidia to existing config..."
    echo 'default-runtime: nvidia' | sudo tee -a /etc/rancher/k3s/config.yaml
  fi
fi

if ! command -v k3s &>/dev/null; then
  echo "Installing K3s..."
  curl -sfL https://get.k3s.io | sh -s -
  sleep 15
else
  echo "K3s already installed: $(k3s --version | head -1)"
  # Verify default_runtime_name is present; if not, config.yaml wasn't picked up
  if ! sudo grep -q "default_runtime_name" /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null; then
    echo "default_runtime_name missing — restarting K3s to pick up config.yaml..."
    sudo systemctl restart k3s
    sleep 15
  fi
fi

# Verify containerd picked up nvidia as default runtime
echo "Verifying containerd nvidia default runtime..."
if sudo grep -q "default_runtime_name" /var/lib/rancher/k3s/agent/etc/containerd/config.toml; then
  sudo grep "default_runtime_name" /var/lib/rancher/k3s/agent/etc/containerd/config.toml
else
  echo "WARNING: default_runtime_name not found in containerd config — GPU pods may not work!"
fi

# Setup kubeconfig
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$USER:$USER" ~/.kube/config
export KUBECONFIG=~/.kube/config
grep -q 'KUBECONFIG' ~/.bashrc || echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc

echo "K3s node status:"
kubectl get nodes

# =============================================================================
# 7. RUNC SYMLINK (required by nvidia-container-runtime)
# =============================================================================
echo "[7/9] Checking runc symlink..."
if [ ! -f /usr/local/sbin/runc ]; then
  echo "Creating runc symlink..."
  sudo ln -sf /var/lib/rancher/k3s/data/current/bin/runc /usr/local/sbin/runc
else
  echo "runc symlink exists: $(runc --version | head -1)"
fi

# =============================================================================
# 8. HELM
# =============================================================================
echo "[8/9] Checking Helm..."
if ! command -v helm &>/dev/null; then
  echo "Installing Helm..."
  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
else
  echo "Helm already installed: $(helm version --short)"
fi

# =============================================================================
# STORAGE DIRECTORIES
# =============================================================================
echo "[+] Setting up /disk directories..."
sudo mkdir -p /disk/if5251/{models,data,pvc,backup,logs,notebooks,datasets,checkpoints}
sudo chown -R "$USER:$USER" /disk/if5251
echo "Storage layout:"
ls -la /disk/if5251/

# =============================================================================
# 9. PYTHON VIRTUAL ENVIRONMENT AND JUPYTERLAB
# =============================================================================
echo "[9/9] Setting up Python environment..."
if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "Installing python3-venv..."
  sudo apt update
  sudo apt install -y python3-venv python3-pip
fi

if [ ! -d "/disk/if5251/venv" ]; then
  echo "Creating virtual environment at /disk/if5251/venv..."
  python3 -m venv /disk/if5251/venv
fi

source /disk/if5251/venv/bin/activate

# Only install packages if not already present
if ! pip show torch &>/dev/null; then
  echo "Installing PyTorch (CUDA 11.8)..."
  pip install --upgrade pip
  pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
else
  echo "PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
fi

if ! pip show jupyterlab &>/dev/null; then
  echo "Installing JupyterLab and ML dependencies..."
  pip install transformers datasets evaluate accelerate \
    jupyterlab pandas scikit-learn matplotlib seaborn \
    scipy ipywidgets
else
  echo "JupyterLab already installed: $(jupyter lab --version)"
fi

# Start JupyterLab in tmux (idempotent)
if ! tmux has-session -t jupyter 2>/dev/null; then
  echo "Starting JupyterLab in tmux session 'jupyter'..."
  tmux new-session -d -s jupyter \
    "source /disk/if5251/venv/bin/activate && jupyter lab \
      --no-browser \
      --port=8888 \
      --ip=0.0.0.0 \
      --notebook-dir=/disk/if5251/notebooks \
      --NotebookApp.password='argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$QyYcDWWOXLCyLXNC9SxHIw\$pg7IKabi/Vq08mpGXXT9uhXHvSwRIqWKfaEktJgf+yc'"
  echo "JupyterLab started. Access at http://<node-ip>:8888"
else
  echo "JupyterLab tmux session already running."
fi

# =============================================================================
# NGINX INGRESS CONTROLLER
# =============================================================================
echo "[+] Deploying NGINX Ingress Controller..."
if ! kubectl get daemonset ingress-nginx-controller -n ingress-nginx &>/dev/null; then
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
  helm repo update

  helm install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx \
    --create-namespace \
    --set controller.service.type=NodePort \
    --set controller.hostNetwork=true \
    --set controller.hostPort.enabled=true \
    --set controller.hostPort.ports.http=80 \
    --set controller.hostPort.ports.https=443 \
    --set controller.kind=DaemonSet \
    --set controller.service.nodePorts.http=30080 \
    --set controller.service.nodePorts.https=30443

  echo "Waiting for ingress-nginx to be ready..."
  kubectl rollout status daemonset/ingress-nginx-controller \
    -n ingress-nginx --timeout=120s
else
  echo "NGINX Ingress Controller already deployed."
fi

# =============================================================================
# NVIDIA DEVICE PLUGIN
# =============================================================================
echo "[+] Deploying NVIDIA device plugin..."
if ! kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &>/dev/null; then
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.19.0/deployments/static/nvidia-device-plugin.yml
  echo "Waiting for device plugin to be ready..."
  kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system --timeout=120s
else
  echo "NVIDIA device plugin already deployed."
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
echo "Bootstrap complete — $(date)"
echo "=========================================="
echo ""
echo "Summary:"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  K3s    : $(kubectl get nodes --no-headers | awk '{print $2}')"
echo "  Helm   : $(helm version --short)"
echo "  Storage: $(df -h /disk | tail -1 | awk '{print $4}') available at /disk"
echo "  GPU cap: $(kubectl get nodes -o json | jq -r '.items[].status.capacity["nvidia.com/gpu"] // "not ready yet"')"
echo ""
echo "Next steps:"
echo "  1. Push this script to GitHub"
echo "  2. Setup K8s namespaces"
echo "  3. Deploy your microservices"