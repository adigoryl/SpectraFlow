#!/bin/bash

# Define variables for all the parameters
DATASET_PATH="./datasets/vasoplegia/CPMG.csv"
SAVE_PATH="./datasets/vasoplegia/preprocessed"
METADATA_COLUMNS=2
UNIQUE_ID_COL="Subject ID"
CLASS_LABEL_COL="Vasoplegia"
#WAVELETS="db1,db7,db20,db38"  # Adjust this list based on your needs
WAVELETS="db1"
SUBS_TO_REMOVE="LPS1,LPS14,LPS19,LPS53,LPS65,LPS141"
STEP_SIZE=0.005
OVERLAP=0.0025
N_COMPONENTS=2

# Directory where the preprocess_control.py script is located relative to the bash script
PREPROCESS_DIR="./preprocessing"

# Execute the Python script with all the parameters
python "$PREPROCESS_DIR/preprocess_control.py" \
  --dataset_path "$DATASET_PATH" \
  --save_path "$SAVE_PATH" \
  --metadata_columns $METADATA_COLUMNS \
  --unique_id_col "$UNIQUE_ID_COL" \
  --class_label_col "$CLASS_LABEL_COL" \
  --wavelets "$WAVELETS" \
  --subs_to_remove "$SUBS_TO_REMOVE" \
  --step_size $STEP_SIZE \
  --overlap $OVERLAP \
  --n_components $N_COMPONENTS
