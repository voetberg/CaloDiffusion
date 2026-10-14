#!/bin/bash
#SBATCH --job-name=layer-calodif
#SBATCH --nodes=1
#SBATCH --account=m2612
#SBATCH --qos regular
#SBATCH --constraint=gpu
#SBATCH --ntasks=1
#SBATCH -G 4
#SBATCH --time=05:00:00
#SBATCH --module=cvmfs
#SBATCH --open-mode=append     # Append output to log files

# Set up environment and DMTCP coordinator
export DMTCP_COORD_HOST=$(hostname)
export DATA_DIR=/global/cfs/cdirs/m2612/calodiffusion

export CONFIG=$HOME/CaloDiffusion/baseline_config.json

export NAME=$(jq -r '.CHECKPOINT_NAME' $HOME/CaloDiffusion/baseline_config.json)

export base_dir=$DATA_DIR/HGCal_showers_william_v2
export TRAIN_DATA=$PSCRATCH/HGCal_showers_william_v2
export RESULTS_DIR=${HOME}/CaloDiffusion/results/${NAME}_Layer

mkdir -p $RESULTS_DIR
mkdir -p $TRAIN_DATA

# Get a list of files in the source directory
files=$(ls "$base_dir")
# Loop through the list of files
for file in $files
do
  # Check if the file exists in the destination directory and only copy it if it does not exist already
  if [ ! -f "$TRAIN_DATA/$file" ]; then
    cp "$base_dir/$file" "$TRAIN_DATA/$file"
    echo "Copied $file to $TRAIN_DATA"
  fi
done

export CHECKPOINT_DIR=$HOME/CaloDiffusion/checkpoints
mkdir -p $CHECKPOINT_DIR

export OUTFILE="$RESULTS_DIR/results.h5"

export BASE_COMMAND="python3 $HOME/CaloDiffusion/calodiffusion/inference.py -c $CONFIG --checkpoint-folder $CHECKPOINT_DIR -d $TRAIN_DATA --hgcal"

export INFERENCE_COMMAND="${BASE_COMMAND} sample --generated $OUTFILE --model-loc ${CHECKPOINT_DIR}/${NAME}_Diffusion/best_val.pth layer --layer-model ${CHECKPOINT_DIR}/${NAME}-layer_LayerModel/best_val.pth"

export PLOT_COMMAND="$BASE_COMMAND plot --generated $OUTFILE --plot-folder ${RESULTS_DIR}/plots/"

module load python
conda activate calodif
$INFERENCE_COMMAND
$PLOT_COMMAND