```bash
cd /home/suchenyu
git clone https://github.com/scy-v/Evo-RL.git
cd Evo-RL
conda create -y -n evo-rl python=3.10 pip ffmpeg=7 -c conda-forge
conda activate evo-rl
export CUDA_VISIBLE_DEVICES=""
uv pip install --default-index https://pypi.tuna.tsinghua.edu.cn/simple -e '.[pi]' 'torch==2.7.1' 'torchvision==0.22.1' 'torchcodec==0.5.0' 'pytest>=8.1,<9' pytest-timeout
```
