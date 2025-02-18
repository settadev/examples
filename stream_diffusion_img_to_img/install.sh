# Steps are from https://github.com/cumulo-autumn/StreamDiffusion

# Exit immediately if any command fails
set -e

pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
pip install setta==0.0.3.dev13