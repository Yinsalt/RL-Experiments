# 1. Conda env (oder venv, was du magst)
conda create -n drl python=3.11 -y
conda activate drl

# 2. PyTorch mit CUDA für deine 3060
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Repo klonen
git clone https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition.git
cd Deep-Reinforcement-Learning-Hands-On-Third-Edition

# 4. Dependencies
pip install -r requirements.txt

# 5. Test ob GPU erkannt wird
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#pip install autorom
#autorom --accept-license
