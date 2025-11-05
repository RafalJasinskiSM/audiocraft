#! /bin/bash
yes | conda create --prefix=.venv python=3.9
conda activate .venv
conda install "ffmpeg<5" -c conda-forge
python -m pip install 'torch==2.1.0'
python -m pip install setuptools wheel
python -m pip install -e '.[wm]'
conda env config vars set AUDIOCRAFT_TEAM=softwaremind
conda deactivate
conda activate .venv/
