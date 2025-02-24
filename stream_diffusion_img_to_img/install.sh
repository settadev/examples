# Steps are from https://github.com/cumulo-autumn/StreamDiffusion

pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121 && \
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt] && \
pip install huggingface-hub==0.25.2 transformers==4.48.3 "numpy<2.0" setta==0.0.7 && \
python -m streamdiffusion.tools.install-tensorrt