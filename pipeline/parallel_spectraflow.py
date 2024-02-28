import os
import random
import argparse
from datetime import datetime
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

# Custom modules (make sure these are correctly imported based on your project structure)
from DataHandler import DataHandler
from FeatureSelector import FeatureSelector
from models.MLPEval import MLPEval
from models.MLPSequentialAttention import SequentialAttentionModel
from LOORFE import LeaveOneOutRecursiveFeatureElimination
from plots.ppm_selected import plot_ppm_with_selection


def parallel_sample_run(args):
    """
    Executes a parallelized feature selection and evaluation experiment based on provided arguments.

    Args:
        args: argparse.Namespace, containing experiment configuration such as dataset path, learning rate,
        number of features to select, batch size, and more.
    """

    mlp_args = {
      "learning_rate": args.learning_rate,
      "decay_steps": args.decay_steps,
      "decay_rate": args.decay_rate,
      "alpha": args.alpha,
      "batch_norm": args.enable_batch_norm,
    }

    fs_args = {
      "num_inputs_to_select": args.num_selected_features,
      "num_inputs_to_select_per_step": (args.num_inputs_to_select_per_step),
    }

    DATA_PRETREATMENTS_MAP = {
        "uv_scaler": {
            "name": "uv_scaler",
            "obj": StandardScaler(),
        },
        "no_scaler": {
            "name": "no_scaler",
            "obj": None,
        },
    }

    # Assuming args.data_pretreatments contains the list of identifiers
    selected_pretreatments = [DATA_PRETREATMENTS_MAP[identifier] for identifier in args.data_pretreatments]

    # Construct save path
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This will go back one directory level
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")

    with Parallel(n_jobs=args.n_jobs, prefer="processes") as parallel:
        for data_candidates in args.data_combs:
            for folds in args.list_of_cv_folds_to_test:
                for pre in selected_pretreatments:
                    pretreatment_name = pre["name"]

                    data_handler = DataHandler(num_classes=2, root=ROOT, dataset_path=args.dataset_path, seed=args.seed)

                    dataset_name, data_folds, subs_in_val_per_sample = data_handler.prepare_data_folds(
                        datasets_to_combine=data_candidates,
                        n_cv_folds=folds,
                        n_cv_repetitions=args.n_cv_repetitions,
                        scaler_obj=pre["obj"],
                    )

                    for epoch in args.list_of_epochs_to_test:
                        SAVE_PATH = ROOT + "/{}/{}_FOLDS_{}_REPS/{}/{}/epoch{}/{}".format(args.experiment_name, folds, args.n_cv_repetitions, now, dataset_name, epoch, pretreatment_name)

                        # Instantiate FeatureSelector
                        fs = FeatureSelector(
                            data_handler=data_handler,
                            model_select_class=SequentialAttentionModel,
                            model_eval_class=MLPEval,
                            fs_args=fs_args,
                            mlp_args=mlp_args,
                            num_epochs_select=epoch,
                            num_epochs_evaluator=args.num_epochs_evaluator,
                            num_selected_features=args.num_selected_features,
                            seed=args.seed
                        )

                        parallel_out = \
                            parallel(
                                (delayed(fs.feature_selection_and_eval)(
                                    data=sample,
                                    batch_size=args.batch_size,
                                    sample_num=str(i + 1),
                                    save_path=os.path.join(SAVE_PATH)
                            ) for i, sample in enumerate(data_folds))
                        )

                        selected_feature_indicies = list(zip(*parallel_out))[0]
                        eval_scores = list(zip(*parallel_out))[1]
                        features_in_selection_order = list(zip(*parallel_out))[2]


                        # Save selected indicies per sample feature selection
                        feature_selection_path = SAVE_PATH + "/selected_feature_indicies.backup"
                        data_handler.pickle_dump(feature_selection_path, selected_feature_indicies)

                        features_in_selection_order_path = SAVE_PATH + "/features_in_selection_order.backup"
                        data_handler.pickle_dump(features_in_selection_order_path, features_in_selection_order)

                        # Compute the mean and standard deviation for each column
                        performance_df = pd.DataFrame(eval_scores)
                        numeric_columns = performance_df.select_dtypes(include=np.number).columns
                        mean_values = performance_df[numeric_columns].mean().round(4)
                        std_values = performance_df[numeric_columns].std().round(4)

                        # Save the mean and standard deviation to a text file
                        with open('{}/results_summary.txt'.format(SAVE_PATH), 'w') as f:
                            f.write('Mean values:\n')
                            f.write(mean_values.to_string())
                            f.write('\n\n')
                            f.write('Standard deviation values:\n')
                            f.write(std_values.to_string())


                        # Add meta data before saving
                        performance_df["dataset"] = dataset_name
                        performance_df["pretreatment"] = pretreatment_name
                        performance_df["epoch"] = epoch

                        # Save performance scores across samples
                        performance_save_path = SAVE_PATH + "/performance_scores.backup"
                        data_handler.pickle_dump(performance_save_path, performance_df)

                        # Save subject in test data partition per sample
                        subs_in_val_per_sample_save_path = SAVE_PATH + "/subs_in_val_per_sample.backup"
                        data_handler.pickle_dump(subs_in_val_per_sample_save_path, subs_in_val_per_sample)

                        # Convert args to dictionary if not already
                        hyperparameters = vars(args)

                        # Now save the hyperparameters to a file
                        with open(f'{SAVE_PATH}/hyperparams.txt', 'w') as f:
                            for key, value in hyperparameters.items():
                                f.write(f'{key}: {value}\n')

                        if dataset_name in ["cpmg_denoised_normalised"]:
                            # Generate ppm chart
                            dataset_dir = "{}/{}/{}.csv".format(ROOT, args.dataset_path, dataset_name)
                            # dataset_dir = ROOT + "/datasets/CPMG/{}.csv".format(dataset_name)
                            ppm_chart_title = "{}, {}_FOLDS_{}_REPS, {}, selected={}, epoch={}, batch={}".format(dataset_name, folds, args.n_cv_repetitions, pretreatment_name, args.num_selected_features, args.num_epochs_select, args.batch_size)
                            top_n_bins_df = plot_ppm_with_selection(dataset_dir, selected_feature_indicies, ppm_chart_title, SAVE_PATH)

                            # Convert the top_N_bins DataFrame to a string
                            top_N_bins_str = top_n_bins_df.to_string(index=False)

                            # Save the string to a text file
                            with open(SAVE_PATH + '/top_N_bins.txt', 'w') as f:
                                f.write(top_N_bins_str)


                        comb_name, Xs, ys = data_handler.load_dataset(data_candidates)
                        loo_rfe = LeaveOneOutRecursiveFeatureElimination(data_handler, MLPEval, num_classes=2, seed=args.seed)

                        loo = LeaveOneOut()

                        parallel_eval = \
                            parallel(
                                (delayed(loo_rfe.evaluate)(
                                    data=(Xs, ys),
                                    train_index=train_index,
                                    test_index=test_index,
                                    batch_size=args.batch_size,
                                    num_selected_features=args.num_selected_features,
                                    sample_num=str(i+1),
                                    subs_in_val_per_sample=subs_in_val_per_sample,
                                    selected_feature_indices=selected_feature_indicies,
                                    scaler_obj=pre["obj"],
                                    save_path=SAVE_PATH,

                                ) for i, (train_index, test_index) in enumerate(loo.split(Xs)))
                            )

                        loo_eval_df = pd.concat(parallel_eval).reset_index(drop=True)
                        # Apply the function to each group
                        loo_rfe_eval_scores_df = loo_eval_df.groupby(['RFE_step']).apply(data_handler.calculate_metrics).reset_index()

                        loo_rfe_eval_path = SAVE_PATH + "/loo_rfe_eval.backup"
                        data_handler.pickle_dump(loo_rfe_eval_path, loo_rfe_eval_scores_df)

                        print("Process Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Feature Selection and Evaluation Experiment")

    # Basic experiment setup
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
    parser.add_argument("--experiment_name", type=str, default="results", help="Experiment name for saving results")
    parser.add_argument("--dataset_path", type=str, default="datasets/vasoplegia/preprocessed", help="Path to the dataset")

    # Dataset and preprocessing
    parser.add_argument('--data_combs', nargs='+', action='append', default=[["cpmg_db1__bin005_overlap0025"]], help="Use data_combs to specify one or more dataset identifiers for the experiment.If a list contains multiple identifiers, those datasets will be merged into a single dataset. Lists with a single identifier are treated as individual datasets.")
    parser.add_argument('--data_pretreatments', nargs='+', choices=['uv_scaler', 'no_scaler'], default=['uv_scaler'], help="List of data pretreatment identifiers")

    # Feature selection and model evaluation parameters
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of jobs to run in parallel")
    parser.add_argument("--list_of_cv_folds_to_test", nargs='+', type=int, default=[10], help="CV folds to test")
    parser.add_argument("--list_of_epochs_to_test", nargs='+', type=int, default=[800], help="Number of epochs for the selection phase (if list: Epochs to test for each fold)")
    parser.add_argument("--num_selected_features", type=int, default=50, help="Number of features to select")
    parser.add_argument("--num_inputs_to_select_per_step", type=int, default=1, help="Number of inputs to select per step")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for the model")
    parser.add_argument("--decay_steps", type=int, default=250, help="Decay steps for the learning rate")
    parser.add_argument("--decay_rate", type=float, default=0.96, help="Decay rate for the learning rate")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha value for L1 regularization")
    parser.add_argument("--enable_batch_norm", action='store_true', help="Enable batch normalization")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs_evaluator", type=int, default=250, help="Number of epochs for the evaluation phase")
    parser.add_argument("--n_cv_repetitions", type=int, default=10, help="Number of cross-validation repetitions")

    args = parser.parse_args()

    # Ensure TensorFlow operations are deterministic
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    parallel_sample_run(args=args)


