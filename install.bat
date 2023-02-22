conda create --name CardiacLab python=3.9 -y  && ^
conda activate CardiacLab && ^
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113  && ^
pip install -r requirements.txt  && ^
PAUSE