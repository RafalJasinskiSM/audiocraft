set -e

# Check if three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 experiment_name solver/config dset/config"
    exit 1
fi

# Pass the arguments to the dora command
conda env config vars set AUDIOCRAFT_DORA_DIR=$1
conda deactivate
conda activate .venv/
dora run solver=$2 dset=$3

