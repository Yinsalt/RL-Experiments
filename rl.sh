# 1. Env erstellen
conda create -n drl python=3.11 -y
conda activate drl

sudo apt-get install git
# 2. PyTorch über conda (mit CUDA)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Was conda hat, über conda
conda install numpy opencv tensorboard pytest -c conda-forge

# 4. Repo klonen
git clone https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition.git
cd Deep-Reinforcement-Learning-Hands-On-Third-Edition

# 5. Rest über pip (nur einmal, am Ende)
pip install gymnasium[atari,classic-control,accept-rom-license] \
    pytorch-ignite ptan stable-baselines3 torchrl "ray[tune]" moviepy mypy
#pip install autorom
#autorom --accept-license
