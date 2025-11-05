#! /bin/bash
conda init
yes | conda create --prefix=.venv python=3.9
conda activate .venv/
yes | conda install "ffmpeg<5" -c conda-forge
yes | python -m pip install 'torch==2.1.0'
yes | python -m pip install setuptools wheel
yes | python -m pip install -e '.[wm]'
conda env config vars set AUDIOCRAFT_TEAM=softwaremind
conda deactivate
conda activate .venv/
