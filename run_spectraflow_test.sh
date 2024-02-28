#!/bin/bash

# run_spectraflow.sh
# Bash script to run the SpectraFlow experiment with specified arguments

# Default values for arguments
seed=2024
experiment_name="results"
dataset_path="datasets/vasoplegia/preprocessed"
data_combs="cpmg_db1__bin005_overlap0025"
data_pretreatments="uv_scaler" # Simplified to just use an identifier
n_jobs=4
list_of_cv_folds_to_test=2
list_of_epochs_to_test=800
num_selected_features=5
num_inputs_to_select_per_step=1
learning_rate=0.0002
decay_steps=250
decay_rate=0.96
alpha=0.01
enable_batch_norm=false # This is handled differently since it's a flag
batch_size=25
num_epochs_evaluator=50
n_cv_repetitions=2

# Collect command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) seed="$2"; shift ;;
        --experiment_name) experiment_name="$2"; shift ;;
        --dataset_path) dataset_path="$2"; shift ;;
        --data_combs) shift; data_combs="$@"; break ;; # Assumes data_combs is the last option or handles as last argument
        --data_pretreatments) data_pretreatments="$2"; shift ;; # Adjusted to handle as a simple string
        --enable_batch_norm) enable_batch_norm=true ;;
        # For flags without a following value, toggle the variable (assuming it's false by default)
    esac
    shift
done

PREPROCESS_DIR="./pipeline"

# Execute the Python script with the collected arguments
python "$PREPROCESS_DIR/parallel_spectraflow.py" \
    --seed "$seed" \
    --experiment_name "$experiment_name" \
    --dataset_path "$dataset_path" \
    --data_combs ${data_combs[@]} \
    --data_pretreatments "$data_pretreatments" \
    --n_jobs "$n_jobs" \
    --list_of_cv_folds_to_test ${list_of_cv_folds_to_test[@]} \
    --list_of_epochs_to_test ${list_of_epochs_to_test[@]} \
    --num_selected_features "$num_selected_features" \
    --num_inputs_to_select_per_step "$num_inputs_to_select_per_step" \
    --learning_rate "$learning_rate" \
    --decay_steps "$decay_steps" \
    --decay_rate "$decay_rate" \
    --alpha "$alpha" \
    ${enable_batch_norm:+--enable_batch_norm} \
    --batch_size "$batch_size" \
    --num_epochs_evaluator "$num_epochs_evaluator" \
    --n_cv_repetitions "$n_cv_repetitions"
