#aneurysm env:
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com

conda create -n aneurysm python=3.8
conda activate aneurysm
pip install opencv-python scikit-learn 
pip install setuptools==58.0.4 
pip install medpy==0.4.0 matplotlib==3.3.4
pip install scikit-image==0.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install nibabel==3.2.2
pip install yacs
pip install torch==1.10.1 torchvision==0.11.2
pip install Pillow==8.4.0 SimpleITK medpy colorlog tensorboardX
pip install numpy==1.19.5

pip cache purge
# or
rm -rf ~/.cache/pip
