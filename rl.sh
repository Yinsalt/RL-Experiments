#!/bin/bash
#===============================================================================
# Deep Reinforcement Learning Hands-On (3rd Edition) - Installationsskript
# Ubuntu 24.04 + RTX 30-Serie + PyTorch 2.5.0 + CUDA 12.4
#===============================================================================
set -e  # Bei Fehler abbrechen

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#===============================================================================
# PHASE 1: Systemvorbereitung und NVIDIA-Treiber
#===============================================================================
phase1_nvidia() {
    log_info "=== PHASE 1: NVIDIA-Treiber ==="
    
    # System aktualisieren
    log_info "System aktualisieren..."
    sudo apt update && sudo apt upgrade -y
    
    # Prüfen ob NVIDIA-GPU vorhanden
    if ! lspci | grep -qi nvidia; then
        log_error "Keine NVIDIA-GPU gefunden!"
        exit 1
    fi
    log_info "GPU gefunden: $(lspci | grep -i nvidia | head -1)"
    
    # Prüfen ob Treiber bereits installiert
    if nvidia-smi &>/dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        log_info "NVIDIA-Treiber bereits installiert: $DRIVER_VERSION"
        
        # Prüfen ob Version >= 525 (minimal für CUDA 12.x)
        MAJOR_VERSION=$(echo $DRIVER_VERSION | cut -d. -f1)
        if [ "$MAJOR_VERSION" -ge 525 ]; then
            log_info "Treiberversion ausreichend, überspringe Installation"
            return 0
        fi
    fi
    
    # Treiber installieren
    log_info "Installiere NVIDIA-Treiber 550..."
    sudo apt install -y ubuntu-drivers-common
    sudo ubuntu-drivers install nvidia:550
    
    log_warn "NEUSTART ERFORDERLICH!"
    log_warn "Nach dem Neustart: ./install_deeprl.sh --phase2"
    exit 0
}

#===============================================================================
# PHASE 2: Miniconda + Python-Umgebung + Pakete
#===============================================================================
phase2_conda() {
    log_info "=== PHASE 2: Conda und Python-Umgebung ==="
    
    # Treiber verifizieren
    if ! nvidia-smi &>/dev/null; then
        log_error "NVIDIA-Treiber nicht geladen! Bitte erst Phase 1 + Neustart"
        exit 1
    fi
    log_info "NVIDIA-Treiber OK: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
    
    # Miniconda installieren falls nicht vorhanden
    CONDA_PATH="$HOME/miniconda3"
    if [ ! -d "$CONDA_PATH" ]; then
        log_info "Installiere Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$CONDA_PATH"
        rm /tmp/miniconda.sh
    else
        log_info "Miniconda bereits installiert"
    fi
    
    # Conda initialisieren für dieses Skript
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    
    # Shell-Integration hinzufügen falls nicht vorhanden
    if ! grep -q "conda initialize" ~/.bashrc; then
        log_info "Füge Conda zu .bashrc hinzu..."
        "$CONDA_PATH/bin/conda" init bash
    fi
    
    # Umgebung erstellen/aktivieren
    ENV_NAME="deeprl"
    if conda env list | grep -q "^$ENV_NAME "; then
        log_warn "Umgebung '$ENV_NAME' existiert bereits. Lösche und erstelle neu..."
        conda env remove -n $ENV_NAME -y
    fi
    
    log_info "Erstelle Conda-Umgebung mit Python 3.11..."
    conda create -n $ENV_NAME python=3.11 -y
    conda activate $ENV_NAME
    
    # PyTorch mit CUDA
    log_info "Installiere PyTorch 2.5.0 mit CUDA 12.4..."
    conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
    
    # Build-Tools für TorchRL
    log_info "Installiere Build-Tools..."
    conda install cmake ninja -c conda-forge -y
    
    # Pip-Pakete
    log_info "Installiere Python-Pakete..."
    pip install --quiet 'numpy<2'
    pip install --quiet opencv-python==4.10.0.84
    pip install --quiet moviepy==1.0.3
    pip install --quiet tensorboard==2.18.0
    pip install --quiet pytorch-ignite==0.5.1
    pip install --quiet mypy==1.8.0
    pip install --quiet pytest
    
    log_info "Installiere Gymnasium mit Atari-Support..."
    pip install --quiet "gymnasium[atari]==0.29.1"
    pip install --quiet "gymnasium[classic-control]==0.29.1"
    pip install --quiet "gymnasium[accept-rom-license]==0.29.1"
    
    log_info "Installiere RL-Bibliotheken..."
    pip install --quiet ptan==0.8.1
    pip install --quiet stable-baselines3==2.3.2
    
    log_info "Installiere TorchRL (kann dauern)..."
    pip install --quiet torchrl==0.6.0
    
    log_info "Installiere Ray Tune (kann dauern)..."
    pip install --quiet "ray[tune]==2.37.0"
    
    # Repository klonen
    REPO_DIR="$HOME/Deep-Reinforcement-Learning-Hands-On-Third-Edition"
    if [ ! -d "$REPO_DIR" ]; then
        log_info "Klone Buch-Repository..."
        git clone --quiet https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition.git "$REPO_DIR"
    else
        log_info "Repository bereits vorhanden"
    fi
    
    log_info "=== PHASE 2 ABGESCHLOSSEN ==="
    log_info "Führe jetzt Verifikation aus: ./install_deeprl.sh --verify"
}

#===============================================================================
# PHASE 3: Verifikation
#===============================================================================
phase3_verify() {
    log_info "=== VERIFIKATION ==="
    
    # Conda laden
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate deeprl
    
    python << 'PYTHON_SCRIPT'
import sys
print("="*60)
print("DEEP RL HANDS-ON - INSTALLATIONSTEST")
print("="*60)

errors = []

# Imports
print("\n[1/6] Teste Imports...")
try:
    import torch, torchvision, gymnasium, numpy, cv2
    import tensorboard, ptan, stable_baselines3, torchrl, ray
    print("      ✓ Alle Imports erfolgreich")
except ImportError as e:
    errors.append(f"Import: {e}")
    print(f"      ✗ {e}")

# CUDA
print("\n[2/6] Teste CUDA...")
try:
    import torch
    assert torch.cuda.is_available(), "CUDA nicht verfügbar"
    print(f"      GPU: {torch.cuda.get_device_name(0)}")
    print(f"      CUDA: {torch.version.cuda}")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print("      ✓ GPU-Berechnung erfolgreich")
except Exception as e:
    errors.append(f"CUDA: {e}")
    print(f"      ✗ {e}")

# Gymnasium Classic
print("\n[3/6] Teste Gymnasium Classic Control...")
try:
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    env.reset()
    env.close()
    print("      ✓ CartPole funktioniert")
except Exception as e:
    errors.append(f"Gymnasium: {e}")
    print(f"      ✗ {e}")

# Gymnasium Atari
print("\n[4/6] Teste Gymnasium Atari...")
try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
    env.reset()
    env.close()
    print("      ✓ Atari/Pong funktioniert")
except Exception as e:
    errors.append(f"Atari: {e}")
    print(f"      ✗ {e}")

# Stable-Baselines3
print("\n[5/6] Teste Stable-Baselines3...")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    env = make_vec_env("CartPole-v1", n_envs=1)
    model = PPO("MlpPolicy", env, verbose=0)
    env.close()
    print("      ✓ PPO-Modell erstellt")
except Exception as e:
    errors.append(f"SB3: {e}")
    print(f"      ✗ {e}")

# Ray
print("\n[6/6] Teste Ray...")
try:
    import ray
    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")
    ray.shutdown()
    print("      ✓ Ray initialisiert")
except Exception as e:
    errors.append(f"Ray: {e}")
    print(f"      ✗ {e}")

# Ergebnis
print("\n" + "="*60)
if errors:
    print(f"FEHLER: {len(errors)} Test(s) fehlgeschlagen")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALLE TESTS ERFOLGREICH! 🎉")
    print("="*60)
    print("\nNächste Schritte:")
    print("  1. Neue Shell öffnen oder: source ~/.bashrc")
    print("  2. conda activate deeprl")
    print("  3. cd ~/Deep-Reinforcement-Learning-Hands-On-Third-Edition")
    print("  4. cd Chapter02 && python 01_cartpole_random.py")
PYTHON_SCRIPT
}

#===============================================================================
# MAIN
#===============================================================================
print_usage() {
    echo "Usage: $0 [--phase1|--phase2|--verify|--full]"
    echo ""
    echo "  --phase1   Nur NVIDIA-Treiber installieren (erfordert Neustart)"
    echo "  --phase2   Conda + Python-Pakete installieren"
    echo "  --verify   Installation verifizieren"
    echo "  --full     Alles in einem (stoppt nach Phase 1 für Neustart)"
    echo ""
    echo "Empfohlener Ablauf:"
    echo "  1. ./install_deeprl.sh --phase1"
    echo "  2. sudo reboot"
    echo "  3. ./install_deeprl.sh --phase2"
    echo "  4. ./install_deeprl.sh --verify"
}

case "${1:-}" in
    --phase1)
        phase1_nvidia
        ;;
    --phase2)
        phase2_conda
        ;;
    --verify)
        phase3_verify
        ;;
    --full)
        phase1_nvidia  # Stoppt automatisch für Neustart falls Treiber installiert
        phase2_conda
        phase3_verify
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
