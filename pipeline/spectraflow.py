# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature selection experiments."""

import os
import random
from datetime import datetime
from joblib import Parallel, delayed
import pickle

from absl import app
from absl import flags
import absl.flags
import pandas as pd
from collections import defaultdict
import copy

from sequential_attention.experiments.datasets.dataset import prepare_data_folds, load_dataset
from sequential_attention.experiments.plots.live_learning import PlotLearning
from sequential_attention.experiments.plots.ppm_selected import plot_ppm_with_selection


from models.MLPSequentialAttention import SequentialAttentionModel
from models.old_models.mlp_sparse import SparseModel

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from typing import List, Tuple
import numpy as np


def jaccard_similarity(list1: List[str], list2: List[str]) -> float:
    """
    Calculate Jaccard Similarity between two lists.

    The Jaccard similarity coefficient measures the size of the intersection divided by the size of the union of two sets.
    The resulting number is a scalar value representing the overall similarity (or stability) of the feature selection
    reflected by the different rankings. Note that the Jaccard similarity ranges from 0 to 1, where 1 signifies that
    the rankings are identical, and 0 signifies that the rankings do not share any features.

    Args:
        list1 (List[str]): First list of features
        list2 (List[str]): Second list of features

    Returns:
        float: Jaccard similarity between the two lists
    """
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def average_jaccard_similarity(rankings: List[List[str]]) -> Tuple[float, float]:
    """
    Calculate the average Jaccard similarity and standard deviation for a list of rankings

    Args:
        rankings (List[List[str]]): List of rankings, where each ranking is a list of features

    Returns:
        Tuple[float, float]: Average Jaccard similarity and standard deviation across all pairs of rankings
    """
    jaccard_similarities = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            similarity = jaccard_similarity(rankings[i], rankings[j])
            jaccard_similarities.append(similarity)
    return np.mean(jaccard_similarities), np.std(jaccard_similarities)


os.environ["TF_DETERMINISTIC_OPS"] = "1"

FLAGS = flags.FLAGS

# Experiment parameters
flags.DEFINE_integer("seed", 2023, "Random seed")
flags.DEFINE_enum("data_name", "mnist", ["cpmg", "mnist", "fashion", "isolet", "mice", "coil", "activity"], "Data name",)
flags.DEFINE_string("model_dir", "/Users/aw678/PycharmProjects/sequential_attention/model_dir", "Checkpoint directory for feature selection model",)

# Feature selection hyperparameters
flags.DEFINE_integer("n_cv_folds", 5, "Number of cross-validation folds")
flags.DEFINE_integer("n_cv_repetitions", 1, "Number of run repetitions")
flags.DEFINE_integer("num_selected_features", 150, "Number of features to select")
flags.DEFINE_enum("algo", "sa", ["sa", "lly", "seql", "gl", "omp"], "Algorithm")
flags.DEFINE_integer("num_inputs_to_select_per_step", 1, "Number of features to select at a time")

# Hyperparameters
flags.DEFINE_float("val_ratio", 0.125, "How much of the training data to split for validation.")
flags.DEFINE_list("deep_layers", "67", "Layers in MLP model")
flags.DEFINE_integer("batch_size", 25, "Batch size")
flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_integer("decay_steps", 250, "Decay steps")
flags.DEFINE_float("decay_rate", 1.0, "Decay rate")
flags.DEFINE_float("alpha", 0.01, "Leaky ReLU alpha")
flags.DEFINE_bool("enable_batch_norm", False, "Enable batch norm")
flags.DEFINE_float("group_lasso_scale", 0.01, "Group LASSO scale")

# Finer control if needed
flags.DEFINE_integer("num_epochs_select", -1, "Number of epochs to fit")
flags.DEFINE_integer("num_epochs_fit", -1, "Number of epochs to select")


# Shallow MLP
def create_model(n_inputs, metrics, num_classes):
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(n_inputs,)))

    # Hidden layer with L1 and L2 regularization and dropout
    model.add(layers.Dense(units=9, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.0005)))
    model.add(layers.Dropout(0.4))

    # Output layer for binary prediction
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    # Compile model
    # old learning rate: learning_rate=0.0015
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model


def reduce_data_to_specific_features(batched_data, selected_indices_list):
    selected_indices_tensor = tf.constant(selected_indices_list, dtype=tf.int32)

    # Function to apply feature selection to a batch
    def select_features(batch_x, batch_y):
        return tf.gather(batch_x, selected_indices_tensor, axis=1), batch_y

    # Map the function over the dataset
    selected_data = batched_data.map(select_features)

    return selected_data


def transform(x, y):
    num_classes = 2
    x = tf.cast(x, dtype=tf.float32)
    return x, tf.one_hot(y, num_classes)


# Save object as pickle file
def pickle_dump(file_path, obj):
    filehandler = open(file_path, 'wb+')
    pickle.dump(obj, filehandler)
    filehandler.close()

# Define a function to calculate metrics
def calculate_metrics(df):
    y_true = df['true_label']
    y_pred = df['y_pred']
    y_score = df['y_pos_pred_probability']

    f1 = f1_score(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)

    # Metrics for positive class
    positive_precision = precision_score(y_true, y_pred, pos_label=1)
    positive_recall = recall_score(y_true, y_pred, pos_label=1)

    # Metrics for negative class
    negative_precision = precision_score(y_true, y_pred, pos_label=0)
    negative_recall = recall_score(y_true, y_pred, pos_label=0)

    return pd.Series({
        'f1': f1,
        'roc_auc': roc_auc,
        'positive_precision': positive_precision,
        'positive_recall': positive_recall,
        'negative_precision': negative_precision,
        'negative_recall': negative_recall
    })


def feature_selection_and_eval(data, batch_size, fs_args, mlp_args, num_epochs_select, num_epochs_fit, sample_num, save_path, num_selected_features, seed, sample_rfe):

    # TODO fix check how does the seed effect the batch shuffling across samples
    tf.random.set_seed(seed)

    # Create save dirs
    sample_save_path = save_path + "/samples/Sample-{}".format(sample_num)
    os.makedirs(sample_save_path, exist_ok=True)

    # eval_model_save_path = sample_save_path + "/model/"
    # os.makedirs(eval_model_save_path, exist_ok=True)

    fs_train_plots_save_path = save_path + "/fs_training_progress_plots"
    os.makedirs(fs_train_plots_save_path, exist_ok=True)

    eval_train_plots_save_path = save_path + "/eval_training_progress_plots"
    os.makedirs(eval_train_plots_save_path, exist_ok=True)

    # The Parallel library doesn't allow to pass tensorflow objects, so we need to batch the data inside the call funct
    x_train_scaled = data["x_train_scaled"]
    y_train = data["y_train"]
    x_val_scaled = data["x_val_scaled"]
    y_val = data["y_val"]
    num_classes = data["num_classes"]
    num_features = data["num_features"]
    is_classification = data["is_classification"]

    # Batch the training data
    buffer_size = x_train_scaled.shape[0]
    ds_train = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train.values))
    ds_train = ds_train.map(transform).shuffle(buffer_size, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)

    # Batch the val data
    ds_val = tf.data.Dataset.from_tensor_slices((x_val_scaled, y_val.values))
    ds_val = ds_val.map(transform)
    ds_val = ds_val.batch(batch_size, drop_remainder=False)

    # Set the missing hyperparameters
    mlp_args["num_classes"] = num_classes
    mlp_args["is_classification"] = is_classification
    fs_args["num_inputs"] = num_features
    fs_args["num_train_steps"] = num_epochs_select * len(ds_train)

    # Concat the two dicts for the feature selection
    args = {**mlp_args, **fs_args}

    # ------------------------------------------
    # ----------- Feature selection ------------
    # ------------------------------------------
    train_plot_title = "FS TRAINING -> sample-{} :: select={}, epoch={}, batch={}".format(sample_num, num_selected_features, num_epochs_select, batch_size)
    save_train_plot_as = fs_train_plots_save_path + "/fs_train_sample-{}.png".format(sample_num)
    callbacks_list = [PlotLearning(train_plot_title, save_train_plot_as)]

    # For positive class:
    positive_precision = tf.keras.metrics.Precision(name='positive_precision', class_id=1)
    positive_recall = tf.keras.metrics.Recall(name='positive_recall', class_id=1)

    # For negative class:
    negative_precision = tf.keras.metrics.Precision(name='negative_precision', class_id=0)
    negative_recall = tf.keras.metrics.Recall(name='negative_recall', class_id=0)

    f1 = tf.keras.metrics.F1Score(average="macro", threshold=None, name='f1_score', dtype=None)
    auc = tf.keras.metrics.AUC(name="auc_score", dtype=None)
    loss_fn = (tf.keras.losses.CategoricalCrossentropy())

    mlp_select = SequentialAttentionModel(**args)
    # mlp_select.compile(loss=loss_fn, metrics=[f1, auc])
    mlp_select.compile(loss=loss_fn, metrics=[f1, auc, positive_precision, positive_recall, negative_precision, negative_recall])
    mlp_select.fit(ds_train, validation_data=ds_val, epochs=num_epochs_select, verbose=2, callbacks=callbacks_list)

    # Get the features
    selected_features = mlp_select.seqatt.selected_features
    _, selected_indices = tf.math.top_k(selected_features, k=num_selected_features)
    selected_indices = selected_indices.numpy()
    selected_features = tf.math.reduce_sum(tf.one_hot(selected_indices, num_features, dtype=tf.int32), 0).numpy()

    eval_plot_title = "EVAL TRAINING -> sample-{} :: select={}, epoch={}, batch={}".format(sample_num, num_selected_features, num_epochs_select, batch_size)
    save_eval_plot_as = eval_train_plots_save_path + "/eval_train_sample-{}.png".format(sample_num)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    eval_callbacks_list = [PlotLearning(eval_plot_title, save_eval_plot_as), early_stop]

    # Decrease the epoch for final model
    mlp_fit = SparseModel(selected_features=selected_features, **mlp_args)

    mlp_fit.compile(loss=loss_fn, metrics=[f1, auc, positive_precision, positive_recall, negative_precision, negative_recall])
    # mlp_fit.compile(loss=loss_fn, metrics=[f1, auc])
    mlp_fit.fit(ds_train, validation_data=ds_val, epochs=num_epochs_fit, verbose=2, callbacks=eval_callbacks_list)

    # print("Finished evaluation model training...")

    # ----------------------------------------------
    # ----------- Eval Model evaluation ------------
    # ----------------------------------------------
    # Load the best performing model
    # mlp_fit.load_weights(eval_model_save_path)
    results_val = mlp_fit.evaluate(ds_val, return_dict=True)

    results = dict()
    results["val_f1"] = round(results_val["f1_score"], 4)
    results["val_loss"] = round(results_val["loss"], 4)
    results["val_auc"] = round(results_val["auc_score"], 4)

    results["val_positive_precision"] = round(results_val["positive_precision"], 4)
    results["val_positive_recall"] = round(results_val["positive_recall"], 4)
    results["val_negative_precision"] = round(results_val["negative_precision"], 4)
    results["val_negative_recall"] = round(results_val["negative_recall"], 4)
    results["sample"] = sample_num

    # Shallow MLP
    ds_train_selected = reduce_data_to_specific_features(ds_train, list(selected_indices))
    ds_val_selected = reduce_data_to_specific_features(ds_val, list(selected_indices))
    # Create and train the model
    # model = create_model(num_selected_features, [f1, auc], num_classes)
    model = create_model(num_selected_features, [f1, auc, positive_precision, positive_recall, negative_precision, negative_recall], num_classes)

    save_eval_plot_as = eval_train_plots_save_path + "/shallow_eval_train_sample-{}.png".format(sample_num)

    model.fit(ds_train_selected,
              epochs=400,
              validation_data=ds_val_selected,
              callbacks=[early_stop, PlotLearning(eval_plot_title, save_eval_plot_as)])

    # Evaluate the model on the test set
    test_loss, test_f1, test_auc, test_positive_precision, test_positive_recall, test_negative_precision, test_negative_recall = model.evaluate(ds_val_selected)
    # print("Test loss:", test_loss)
    # print("Test F1 score:", test_f1)
    # print("Test AUC score:", test_auc)

    results["val_f1_shallow"] = round(test_f1, 4)
    results["val_loss_shallow"] = round(test_loss, 4)
    results["val_auc_shallow"] = round(test_auc, 4)

    results["val_positive_precision_shallow"] = round(test_positive_precision, 4)
    results["val_positive_recall_shallow"] = round(test_positive_recall, 4)
    results["val_negative_precision_shallow"] = round(test_negative_precision, 4)
    results["val_negative_recall_shallow"] = round(test_negative_recall, 4)

    # Once the feature selection is performed, recrusively eliminate two worst performing and check performance
    features_in_selection_order = mlp_select.get_feature_selection_order().numpy().copy()


    return list(selected_indices), results,  mlp_select.get_feature_selection_order().numpy()



def leave_one_out_eval(data, train_index, test_index, batch_size, num_selected_features, sample_num, subs_in_val_per_sample, selected_feature_indicies, scaler_obj, seed, save_path, num_classes=2):
    tf.random.set_seed(seed)

    def get_keys_where_value_exists(dict_list, X):
        """
        Get the keys of all dictionaries within a list where X is within the list of values.

        Parameters:
        dict_list (list of dict): List of dictionaries.
        X (int): The value to be checked.

        Returns:
        keys (list of str): List of keys where X is in their associated values.
        """

        # Initialize an empty list to store keys
        keys = []

        # Iterate over the dictionaries in the list
        for dict in dict_list:
            # Iterate over each item in the dictionary
            for key, values in dict.items():
                # Check if X is in the values
                if X in values:
                    # If X is in the values, add the key to the list
                    keys.append(key)

        return keys

    def get_top_features(ordered_rankings, ascending=False, n_features=None):
        """
        Extract top features from ordered rankings based on frequency and mean position.

        Parameters:
        ordered_rankings (list of lists): Ordered feature rankings.
        n_features (int, optional): Number of top features to select. If not provided, all unique features are selected.

        Returns:
        top_features (DataFrame): DataFrame containing top features and their associated stats.
        """

        # Copy the input to avoid changing it outside the function
        data = copy.deepcopy(ordered_rankings)

        # Get unique features
        unique_features = list(set(item for sublist in data for item in sublist))

        # If n_features is not provided, set it to the number of unique features
        if n_features is None:
            n_features = len(unique_features)

        # Prepare a dict to hold index positions of each unique feature
        index_positions = defaultdict(list)
        for i, sublist in enumerate(data):
            for j, item in enumerate(sublist):
                index_positions[item].append(j + 1)  # add 1 to the index

        # Now calculate the statistics for index positions
        # Create a DataFrame to store the stats
        df = pd.DataFrame(index=unique_features,
                          columns=['mean', 'max', 'min', 'Q1', 'Q3', 'frequency'])

        # Iterate over each unique feature
        for feature in unique_features:
            positions = index_positions[feature]
            df.loc[feature, 'mean'] = np.mean(positions)
            df.loc[feature, 'max'] = np.max(positions)
            df.loc[feature, 'min'] = np.min(positions)
            df.loc[feature, 'Q1'] = np.percentile(positions, 25)
            df.loc[feature, 'Q3'] = np.percentile(positions, 75)
            df.loc[feature, 'frequency'] = len(positions)

        # Convert 'frequency' column to integer
        df['frequency'] = df['frequency'].astype(int)

        # Select the top features based on the frequency
        top_features = df.nlargest(n_features, 'frequency')

        # Order the selected features based on the mean in descending order
        top_features = top_features.sort_values(by='mean', ascending=ascending)

        return top_features

    Xs, ys = data
    # Get feature selection rankings where the "test_index" subject wasn't a part of the training data
    indexes = get_keys_where_value_exists(subs_in_val_per_sample, test_index)
    unbiased_rankings = [selected_feature_indicies[i] for i in indexes]

    # Merge the rankings into one accordingly to selection frequency and mean position
    top_ranking = get_top_features(ordered_rankings=unbiased_rankings, ascending=True, n_features=num_selected_features)

    # Select the data accordingly to the split
    x_train, x_test = Xs.iloc[train_index], Xs.iloc[test_index]
    y_train, y_test = ys.iloc[train_index], ys.iloc[test_index]

    # Scale the data using the provided scaler, if any
    if scaler_obj is not None:
        scaler = copy.deepcopy(scaler_obj)
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    else:
        x_train_scaled = copy.deepcopy(x_train)
        x_test_scaled = copy.deepcopy(x_test)


    # Batch the training data
    buffer_size = x_train_scaled.shape[0]
    ds_train = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train.values))
    ds_train = ds_train.map(transform).shuffle(buffer_size, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)

    # Batch the val data
    ds_test = tf.data.Dataset.from_tensor_slices((x_test_scaled, y_test.values))
    ds_test = ds_test.map(transform)
    ds_test = ds_test.batch(batch_size, drop_remainder=False)


    # Define the scoring metrics

    f1 = tf.keras.metrics.F1Score(average="macro", threshold=None, name='f1_score', dtype=None)
    auc = tf.keras.metrics.AUC(name="auc_score", dtype=None)

    features_in_selection_order = top_ranking.index

    loo_rfe_res = []

    while len(features_in_selection_order) > 1:
        ds_train_selected = reduce_data_to_specific_features(ds_train, list(features_in_selection_order))
        ds_test_selected = reduce_data_to_specific_features(ds_test, list(features_in_selection_order))
        # Create and train the model
        model = create_model(len(features_in_selection_order), [f1, auc], num_classes)

        loo_rfe_eval_train_plots_save_path = save_path + "/loo_rfe_eval_training_progress_plots/{}_features".format(len(features_in_selection_order))
        os.makedirs(loo_rfe_eval_train_plots_save_path, exist_ok=True)
        save_eval_plot_as = loo_rfe_eval_train_plots_save_path + "/loo_rfe_eval_train_sample-{}.png".format(sample_num)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        model.fit(
            ds_train_selected,
            epochs=400,
            validation_data=ds_test_selected,
            callbacks=[early_stop, PlotLearning(save_eval_plot_as, save_eval_plot_as)]
        )

        # Evaluate the model on the test set
        rfe_test_loss, rfe_test_f1, rfe_test_auc = model.evaluate(ds_test_selected)

        rfe_results = {}
        rfe_results["rfe_val_loss_shallow"] = round(rfe_test_loss, 6)
        rfe_results["subject_num"] = sample_num
        rfe_results["RFE_step"] = len(features_in_selection_order)

        # Get the probabilities of positive class
        y_pred = model.predict(ds_test_selected)

        # model returns probabilities for both classes (0 and 1). So we use the probabilities for the positive class (usually class 1) when calculating the ROC AUC score later
        rfe_results["y_pos_pred_probability"] = y_pred[0][1]
        rfe_results["true_label"] = y_test.item()

        # Get the index of the highest value in an array as the predicted label
        rfe_results["y_pred"] = np.argmax(y_pred, axis=1).item()

        loo_rfe_res.append(rfe_results)

        # Remove features that were selected last and repeat
        features_in_selection_order = features_in_selection_order[:-2]

    return pd.DataFrame(loo_rfe_res)


def parallel_sample_run(test_folds_list, test_epochs_list, data_combs, data_pretreatments, batch_size=256, num_epochs_select=250, num_epochs_fit=250, learning_rate=0.0002, decay_steps=100, decay_rate=1.0, model_dir=None, num_selected_features=None, seed=1, sample_rfe=False):
    """Run a feature selection experiment with a given set of hyperparameters."""

    mlp_args = {
      "layer_sequence": [int(i) for i in FLAGS.deep_layers],
      "learning_rate": learning_rate,
      "decay_steps": decay_steps,
      "decay_rate": decay_rate,
      "alpha": FLAGS.alpha,
      "batch_norm": FLAGS.enable_batch_norm,
    }

    fs_args = {
      "num_inputs_to_select": FLAGS.num_selected_features,
      "num_inputs_to_select_per_step": (FLAGS.num_inputs_to_select_per_step),
    }


    # data = get_dataset(
    #     data_name=FLAGS.data_name,
    #     n_cv_folds=FLAGS.n_cv_folds,
    #     n_cv_repetitions=FLAGS.n_cv_repetitions,
    #     batch_size=batch_size,
    #     seed=FLAGS.seed,
    # )

    # Construct save path
    ROOT = os.path.dirname(os.path.abspath(__file__))
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")

    n_jobs = 90
    with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
        dataset_path = "datasets/vasoplegia/different_wavelets"
        # dataset_path = "datasets/vasoplegia/"

        for data_candidates in data_combs:
            for folds in test_folds_list:
                for pre in data_pretreatments:
                    pretreatment_name = pre["name"]

                    dataset_name, data_folds, subs_in_val_per_sample = prepare_data_folds(
                        datasets_to_combine=data_candidates,
                        root=ROOT,
                        dataset_path=dataset_path,
                        n_cv_folds=folds,
                        n_cv_repetitions=FLAGS.n_cv_repetitions,
                        scaler_obj=pre["obj"],
                        seed=FLAGS.seed,
                    )

                    for epoch in test_epochs_list:

                        SAVE_PATH = ROOT + "/More_metrics/CPMG_untargeted/{}_FOLDS_{}_REPS/{}/{}/epoch{}/{}".format(folds, FLAGS.n_cv_repetitions, now, dataset_name, epoch, pretreatment_name)

                        parallel_out = \
                            parallel(
                                (delayed(feature_selection_and_eval)(
                                    data=sample,
                                    batch_size=batch_size,
                                    fs_args=fs_args,
                                    mlp_args=mlp_args,
                                    num_epochs_select=epoch,
                                    num_epochs_fit=num_epochs_fit,
                                    sample_num=str(i+1),
                                    save_path=SAVE_PATH,
                                    num_selected_features=num_selected_features,
                                    seed=seed,
                                    sample_rfe=sample_rfe

                                ) for i, sample in enumerate(data_folds))
                            )

                        selected_feature_indicies = list(zip(*parallel_out))[0]
                        eval_scores = list(zip(*parallel_out))[1]
                        if sample_rfe:
                            rfe_eval_scores = list(zip(*parallel_out))[2]
                            features_in_selection_order = list(zip(*parallel_out))[3]
                        else:
                            features_in_selection_order = list(zip(*parallel_out))[2]


                        # Save selected indicies per sample feature selection
                        feature_selection_path = SAVE_PATH + "/selected_feature_indicies.backup"
                        pickle_dump(feature_selection_path, selected_feature_indicies)

                        features_in_selection_order_path = SAVE_PATH + "/features_in_selection_order.backup"
                        pickle_dump(features_in_selection_order_path, features_in_selection_order)

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
                        pickle_dump(performance_save_path, performance_df)

                        if sample_rfe:
                            # Save RFE performance scores
                            df_rfe_performance_scores = pd.concat(rfe_eval_scores).reset_index(drop=True)
                            # Add meta data before saving
                            df_rfe_performance_scores["dataset"] = dataset_name
                            df_rfe_performance_scores["pretreatment"] = pretreatment_name
                            df_rfe_performance_scores["epoch"] = epoch

                            rfe_performance_save_path = SAVE_PATH + "/rfe_performance_scores.backup"
                            pickle_dump(rfe_performance_save_path, df_rfe_performance_scores)

                        # Save subject in test data partition per sample
                        subs_in_val_per_sample_save_path = SAVE_PATH + "/subs_in_val_per_sample.backup"
                        pickle_dump(subs_in_val_per_sample_save_path, subs_in_val_per_sample)

                        # Save hyperparameters
                        flags = absl.flags.FLAGS
                        values = flags.flag_values_dict()
                        with open(SAVE_PATH + '/hyperparams.txt', 'w') as f:
                            for key, value in values.items():
                                f.write(f'{key}: {value}\n')

                        if dataset_name in ["cpmg_denoised_normalised"]:
                            # Generate ppm chart
                            dataset_dir = "{}/{}/{}.csv".format(ROOT, dataset_path, dataset_name)
                            # dataset_dir = ROOT + "/datasets/CPMG/{}.csv".format(dataset_name)
                            ppm_chart_title = "{}, {}_FOLDS_{}_REPS, {}, selected={}, epoch={}, batch={}".format(dataset_name, folds, FLAGS.n_cv_repetitions, pretreatment_name, num_selected_features, num_epochs_select, batch_size)
                            top_n_bins_df = plot_ppm_with_selection(dataset_dir, selected_feature_indicies, ppm_chart_title, SAVE_PATH)

                            # Convert the top_N_bins DataFrame to a string
                            top_N_bins_str = top_n_bins_df.to_string(index=False)

                            # Save the string to a text file
                            with open(SAVE_PATH + '/top_N_bins.txt', 'w') as f:
                                f.write(top_N_bins_str)


                        comb_name, Xs, ys = load_dataset(data_candidates, ROOT, dataset_path)

                        loo = LeaveOneOut()

                        parallel_eval = \
                            parallel(
                                (delayed(leave_one_out_eval)(
                                    data=(Xs, ys),
                                    train_index=train_index,
                                    test_index=test_index,
                                    batch_size=batch_size,
                                    num_selected_features=num_selected_features,
                                    sample_num=i,
                                    subs_in_val_per_sample=subs_in_val_per_sample,
                                    selected_feature_indicies=selected_feature_indicies,
                                    scaler_obj=pre["obj"],
                                    seed=seed,
                                    save_path=SAVE_PATH,

                                ) for i, (train_index, test_index) in enumerate(loo.split(Xs)))
                                # ) for i, (train_index, test_index) in enumerate(itertools.islice(loo.split(Xs), 5)))
                            )

                        loo_eval_df = pd.concat(parallel_eval).reset_index(drop=True)
                        # Apply the function to each group
                        loo_rfe_eval_scores_df = loo_eval_df.groupby(['RFE_step']).apply(calculate_metrics).reset_index()

                        loo_rfe_eval_path = SAVE_PATH + "/loo_rfe_eval.backup"
                        pickle_dump(loo_rfe_eval_path, loo_rfe_eval_scores_df)

                        print("Process Finished")


    return None


def main(args):
    del args  # Not used.

    os.environ["PYTHONHASHSEED"] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    tf.keras.backend.clear_session()

    num_epochs_select = FLAGS.num_epochs
    num_epochs_fit = 400

    DATASET_COMBINATIONS = [
        ["cpmg_db1__pca_reduction__bin005_overlap0025"],
        ["cpmg_db2__pca_reduction__bin005_overlap0025"],
    ]

    DATA_PRETREATMENTS = [
        {
            "name": "auto_scaler",
            "obj": StandardScaler()
        },
        # {
        #     "name": "no_scaler",
        #     "obj": None
        # },
    ]

    # test_folds_list = [4, 6, 8, 10]
    test_folds_list = [10]

    test_epochs_list = [500, 800, 1100]
    # test_epochs_list = [800]

    # test_epochs_list = list(range(300, 7800, 300))


    parallel_sample_run(
        test_folds_list=test_folds_list,
        test_epochs_list=test_epochs_list,
        data_combs=DATASET_COMBINATIONS,
        data_pretreatments=DATA_PRETREATMENTS,
        batch_size=FLAGS.batch_size,
        # num_epochs_select=num_epochs_select,
        num_epochs_fit=num_epochs_fit,
        learning_rate=FLAGS.learning_rate,
        decay_steps=FLAGS.decay_steps,
        decay_rate=FLAGS.decay_rate,
        model_dir=FLAGS.model_dir,
        num_selected_features=FLAGS.num_selected_features,
        seed=FLAGS.seed,
        sample_rfe=True
    )


if __name__ == "__main__":
  app.run(main)



# def reduce_data_to_specific_features(batched_data, selected_indices_list):
#     selected_indices_tensor = tf.constant(selected_indices_list, dtype=tf.int32)
#
#     # Function to apply feature selection to a batch
#     def select_features(batch_x, batch_y):
#         return tf.gather(batch_x, selected_indices_tensor, axis=1), batch_y
#
#     # Map the function over the dataset
#     selected_data = batched_data.map(select_features)
#
#     return selected_data
#
#
# ds_train_selected = reduce_data_to_specific_features(ds_train, list(selected_indices))
# ds_val_selected = reduce_data_to_specific_features(ds_val, list(selected_indices))

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import regularizers
# from kerastuner import RandomSearch
# from kerastuner import Objective
#
# f1 = tf.keras.metrics.F1Score(average="macro", threshold=None, name='f1_score', dtype=None)
# auc = tf.keras.metrics.AUC(name="auc_score", dtype=None)
#
# def build_model(hp):
#     model = keras.Sequential()
#
#     # Input layer
#     model.add(layers.Input(shape=(num_features,)))  # Update the shape
#
#     # Hidden layer with L1 and L2 regularization and dropout
#     model.add(layers.Dense(units=hp.Int('units', min_value=10, max_value=40, step=1),
#                            activation='relu',
#                            kernel_regularizer=regularizers.l1_l2(
#                                l1=hp.Float('l1', min_value=1e-5, max_value=1e-2, sampling='LOG'),
#                                l2=hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='LOG'))))
#     model.add(layers.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
#
#     # Output layer for binary prediction
#     model.add(layers.Dense(num_classes, activation='sigmoid'))  # Update the number of output nodes
#
#     # Compile model
#     model.compile(
#         optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
#         loss='binary_crossentropy',
#         metrics=[f1, auc])
#
#     return model
#
# # Define the Keras Tuner search space
# tuner = RandomSearch(
#     build_model,
#     objective=Objective("val_f1_score", direction="max"),  # Update the objective
#     max_trials=50,
#     executions_per_trial=3,
#     directory="/Users/aw678/PycharmProjects/sequential_attention/model_dir/tmp",
#     project_name='binary_prediction',
# )
#
# # Search for the best hyperparameters
# tuner.search_space_summary()
#
# # Perform the hyperparameter search
# tuner.search(ds_train,  # Use the dataset instead of X_train and y_train
#              epochs=100,
#              validation_data=ds_val,  # Use the dataset instead of validation_split
#              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
#
# # Get the optimal hyperparameters
# best_hp = tuner.get_best_hyperparameters()[0]
#
# # Print the best hyperparameters
# print("Best Hyperparameters:")
# print(best_hp.values)
#
# # Train the model with the best hyperparameters
# best_model = tuner.hypermodel.build(best_hp)
# best_model.fit(ds_train,
#                epochs=100,
#                validation_data=ds_val,
#                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
#
# # Evaluate the model on the test set
# test_loss, test_f1, test_auc = best_model.evaluate(ds_val)
# print("Test loss:", test_loss)
# print("Test F1 score:", test_f1)
# print("Test AUC score:", test_auc)


# ----------------------------------------------------
# # Fetch all the results and print
# ------------------------------------------
# def get_val_f1_shallow(file_path):
#     mean_values = False
#
#     val_f1 = None
#     val_auc = None
#     val_f1_shallow = None
#     val_auc_shallow = None
#
#     with open(file_path, 'r') as file:
#         for line in file.readlines():
#             if "Mean values" in line:
#                 mean_values = True
#             elif "Standard deviation values" in line:
#                 mean_values = False
#
#             if mean_values:
#                 if line.startswith("val_f1 "):
#                     val_f1 = float(line.split()[-1])
#                 elif line.startswith("val_auc "):
#                     val_auc = float(line.split()[-1])
#                 elif line.startswith("val_f1_shallow"):
#                     val_f1_shallow = float(line.split()[-1])
#                 elif line.startswith("val_auc_shallow"):
#                     val_auc_shallow = float(line.split()[-1])
#
#     return val_f1, val_auc, val_f1_shallow, val_auc_shallow
#
#
# a = "/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/Test"
# path_list = glob.glob(a + "/*/*/*/*/*/results_summary.txt")
# largest_val = 0
# largest_path = ""
# for path in path_list:
#     path_components = path.split(os.sep)
#
#     # if "10_REPS" in path_components[8]:
#     #     # if path_components[8] == "2023-04-12_16:15":
#     #     if path_components[8] == "2023-05-03_12:52":
#     a, b, c, d = get_val_f1_shallow(path)
#     print("{}/{}/{}: f1 {} auc {} | f1 {} auc {}".format(path_components[8], path_components[10],
#                                                          path_components[11], a, b, c, d))

# ----------------------------------------------------------
# PLOT THE SCORE AND SIMILARTY PER EPOCH, OR X EPOCH PER FEATURE SELECTED
# ----------------------

# import pandas as pd
# import plotly.graph_objects as go
# import plotly.offline
# import glob
# import os
# import re
#
# def extract_values(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()
#
#     values = content.split('Standard deviation values:')[1].strip().split('\n')
#     mean_values = content.split('Standard deviation values:')[0].split('Mean values:')[1].strip().split('\n')
#
#     mean_values_dict = dict((item.split()[0], float(item.split()[1])) for item in mean_values)
#     std_dev_values_dict = dict((item.split()[0], float(item.split()[1])) for item in values)
#
#     return mean_values_dict['val_f1_shallow'], mean_values_dict['val_auc_shallow'], std_dev_values_dict[
#         'val_f1_shallow'], std_dev_values_dict['val_auc_shallow']
#
#
# def sort_paths_by_epoch(paths):
#     def extract_epoch_number(path):
#         match = re.search(r'epoch(\d+)', path)
#         return int(match.group(1)) if match else float('inf')
#
#     return sorted(paths, key=extract_epoch_number)
#
#
# def jaccard_similarity(list1: List[str], list2: List[str]) -> float:
#     """
#     Calculate Jaccard Similarity between two lists.
#
#     The Jaccard similarity coefficient measures the size of the intersection divided by the size of the union of two sets.
#     The resulting number is a scalar value representing the overall similarity (or stability) of the feature selection
#     reflected by the different rankings. Note that the Jaccard similarity ranges from 0 to 1, where 1 signifies that
#     the rankings are identical, and 0 signifies that the rankings do not share any features.
#
#     Args:
#         list1 (List[str]): First list of features
#         list2 (List[str]): Second list of features
#
#     Returns:
#         float: Jaccard similarity between the two lists
#     """
#     intersection = len(list(set(list1).intersection(list2)))
#     union = (len(list1) + len(list2)) - intersection
#     return float(intersection) / union
#
#
# def average_jaccard_similarity(rankings: List[List[str]]) -> Tuple[float, float]:
#     """
#     Calculate the average Jaccard similarity and standard deviation for a list of rankings
#
#     Args:
#         rankings (List[List[str]]): List of rankings, where each ranking is a list of features
#
#     Returns:
#         Tuple[float, float]: Average Jaccard similarity and standard deviation across all pairs of rankings
#     """
#     jaccard_similarities = []
#     for i in range(len(rankings)):
#         for j in range(i + 1, len(rankings)):
#             similarity = jaccard_similarity(rankings[i], rankings[j])
#             jaccard_similarities.append(similarity)
#     return round(np.mean(jaccard_similarities), 4), round(np.std(jaccard_similarities), 4)
#
#
# a = "/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/Test"
# path_list = glob.glob(a + "/*/*/*/*/*/results_summary.txt")
# path_list = path_list[0:6]
# sorted_paths = sort_paths_by_epoch(path_list)
#
# result_lists = {
#     'mean_f1': [],
#     'mean_auc': [],
#     'std_f1': [],
#     'std_auc': [],
#     'avg_similarity': [],
#     'std_similarity': [],
#     'epoch_per_selected_feature': []
# }
#
# for path in sorted_paths:
#     path_components = path.split(os.sep)
#
#     mean_f1, mean_auc, std_f1, std_auc = extract_values(path)
#     result_lists['mean_f1'].append(mean_f1)
#     result_lists['mean_auc'].append(mean_auc)
#     result_lists['std_f1'].append(std_f1)
#     result_lists['std_auc'].append(std_auc)
#
#     # print("{}/{}/{}: mean f1 {} mean auc {} | std f1 {} std auc {}".format(path_components[8], path_components[10],
#     #                                                      path_components[11], a, b, c, d))
#
#     # Get the directory part of the old path
#     directory = os.path.dirname(path)
#
#     # Define the new file name
#     new_file_name = "selected_feature_indicies.backup"
#
#     # Join the directory with the new file name
#     rankings_path = os.path.join(directory, new_file_name)
#
#     with open(rankings_path, 'rb') as f:
#         rankings = pickle.load(f)
#
#     avg_similarity, std_dev = average_jaccard_similarity(rankings)
#     result_lists['avg_similarity'].append(avg_similarity)
#     result_lists['std_similarity'].append(std_dev)
#
#     epoch = int(re.findall(r'\d+', path_components[11])[0])
#     num_of_features_to_select = 50
#     result_lists['epoch_per_selected_feature'].append(epoch // num_of_features_to_select)
#
# # # Now you have your separate lists:
# # print(result_lists['mean_f1'])
# # print(result_lists['mean_auc'])
# # print(result_lists['std_f1'])
# # print(result_lists['std_auc'])
# # print(result_lists['avg_similarity'])
# # print(result_lists['std_similarity'])
# # print(result_lists['epoch_per_selected_feature'])
#
#
#
#
# def add_trace_with_std(fig, df, x, y, y_std, name, color, yaxis='y'):
#     lighter_color = [min(c + 50, 255) for c in color]
#     darker_color = [max(c - 50, 0) for c in color]
#
#     fig.add_trace(go.Scatter(x=df[x], y=df[y] + df[y_std], mode='lines',
#                              line=dict(color=f'rgba({lighter_color[0]}, {lighter_color[1]}, {lighter_color[2]}, 0.3)'),
#                              showlegend=False, yaxis=yaxis, name=name))
#
#     fig.add_trace(go.Scatter(x=df[x], y=df[y] - df[y_std], mode='lines',
#                              line=dict(color=f'rgba({lighter_color[0]}, {lighter_color[1]}, {lighter_color[2]}, 0.3)'),
#                              fill='tonexty', showlegend=False, yaxis=yaxis, name=name))
#
#     fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines+markers',
#                              line=dict(color=f'rgb({darker_color[0]}, {darker_color[1]}, {darker_color[2]})'),
#                              name=name, yaxis=yaxis))
#
#
# df = pd.DataFrame(result_lists)
#
# fig = go.Figure()
#
# add_trace_with_std(fig, df, 'epoch_per_selected_feature', 'mean_auc', 'std_auc', 'Mean AUC', [31, 119, 180])
# add_trace_with_std(fig, df, 'epoch_per_selected_feature', 'avg_similarity', 'std_similarity', 'Avg Similarity', [255, 127, 14], 'y2')
#
# fig.update_layout(
#     xaxis=dict(title='Epoch per selected feature'),
#     yaxis=dict(title='Mean AUC'),
#     yaxis2=dict(
#         title='Avg Similarity',
#         overlaying='y',
#         side='right'
#     ),
#     template='plotly_white'
# )
#
# plotly.offline.plot(fig, filename='plot.html')




# ----------------------------------------
# PLOT THE PREDICTION ACCURACY AND SIMILARITY FOR RECURSIVE FEATURE ELIMINATION AFTER SEQUENTIAL ATTENTION FEATURE SELECTION
# ---------------------------------

# import pandas as pd
# import plotly.graph_objects as go
#
# # Join the directory with the new file name
# path = "/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/RFE_seq_attention/10_FOLDS_10_REPS/2023-05-26_20:00/CPMG/epoch1500/auto_scaler/rfe_performance_scores.backup"
#
# with open(path, 'rb') as f:
#     df = pickle.load(f)
#
# # assuming df is your DataFrame
# stats_df = df.groupby('RFE_step')['rfe_val_auc_shallow'].agg(['mean', 'std'])
#
# path = "/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/RFE_seq_attention/10_FOLDS_10_REPS/2023-05-26_20:00/cpmg_denoised__pca_reduction__bin005_overlap0025/epoch1500/auto_scaler/features_in_selection_order_path.backup"
# with open(path, 'rb') as f:
#     ordered_ranking = pickle.load(f)
#
# sins_list = []
# for i in list(range(len(ordered_ranking[0]), 10, -2)):
#     a = [arr[0:i] for arr in ordered_ranking]
#     avg_similarity, std_dev = average_jaccard_similarity(a)
#
#     sins_list.append({
#         "avg_similarity": avg_similarity,
#         "std_dev": std_dev,
#         "RFE_step": i
#     })
#
# sims_df = pd.DataFrame(sins_list)
#
#
# def add_trace(fig, x, y, y_upper, y_lower, name, color, fillcolor, yaxis):
#     fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color), yaxis=yaxis))
#     fig.add_trace(
#         go.Scatter(x=x, y=y_upper, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fillcolor, yaxis=yaxis,
#                    showlegend=False))
#     fig.add_trace(
#         go.Scatter(x=x, y=y_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fillcolor, yaxis=yaxis,
#                    showlegend=False))
#
#
# # Create a new figure
# fig = go.Figure()
#
# # List of data to add to the plot
# data_to_plot = [
#     {
#         'x': stats_df.index,
#         'y': stats_df['mean'],
#         'y_upper': stats_df['mean'] + stats_df['std'],
#         'y_lower': stats_df['mean'] - stats_df['std'],
#         'name': 'Mean Test Accuracy',
#         'color': 'blue',
#         'fillcolor': 'rgba(0, 0, 255, 0.2)',
#         'yaxis': 'y1'
#     },
#     {
#         'x': sims_df['RFE_step'],
#         'y': sims_df['avg_similarity'],
#         'y_upper': sims_df['avg_similarity'] + sims_df['std_dev'],
#         'y_lower': sims_df['avg_similarity'] - sims_df['std_dev'],
#         'name': 'Mean Ranking Similarity',
#         'color': 'red',
#         'fillcolor': 'rgba(255, 0, 0, 0.2)',
#         'yaxis': 'y2'
#     }
# ]
#
# # Add data to the plot
# for data in data_to_plot:
#     add_trace(fig, **data)
#
# # Update layout to include a secondary Y-axis
# fig.update_layout(
#     title='Prediction accuracy and ranking similarity across samples.',
#     xaxis_title='Number of Features',
#     yaxis_title='AUC Accuracy',
#     yaxis2=dict(title='Mean Ranking Similarity', overlaying='y', side='right')
# )
#
# plotly.offline.plot(fig, filename='plot.html')



# --------------------------------------------------------
# FEATURE RANKING PLOT
# ---------
#
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# import plotly.graph_objects as go
# import copy
#
# # Assuming data is your list of rankings
# data = copy.deepcopy(ordered_ranking)
#
# # Get unique features
# unique_features = list(set(item for sublist in data for item in sublist))
#
# # Prepare a dict to hold index positions of each unique feature
# index_positions = defaultdict(list)
# for i, sublist in enumerate(data):
#     for j, item in enumerate(sublist):
#         index_positions[item].append(j + 1)  # add 1 to the index
#
# # Now calculate the statistics for index positions
# # Create a DataFrame to store the stats
# df = pd.DataFrame(index=unique_features, columns=['mean', 'max', 'min', 'Q1', 'Q3', 'frequency'])
#
# # Iterate over each unique feature
# for feature in unique_features:
#     positions = index_positions[feature]
#     df.loc[feature, 'mean'] = np.mean(positions)
#     df.loc[feature, 'max'] = np.max(positions)
#     df.loc[feature, 'min'] = np.min(positions)
#     df.loc[feature, 'Q1'] = np.percentile(positions, 25)
#     df.loc[feature, 'Q3'] = np.percentile(positions, 75)
#     df.loc[feature, 'frequency'] = len(positions)
#
# # Convert 'frequency' column to integer
# df['frequency'] = df['frequency'].astype(int)
#
# # Select the top 50 features based on the frequency
# top_features = df.nlargest(50, 'frequency')
#
# # Order the selected features based on the mean in descending order
# top_features = top_features.sort_values(by='mean', ascending=False)
#
# # Create a horizontal box plot
# fig = go.Figure()
#
# for feature in top_features.index:
#     # Append the frequency to the feature name
#     feature_name = f"{feature} (frq{top_features.loc[feature, 'frequency']})"
#     fig.add_trace(go.Box(x=index_positions[feature], name=feature_name, orientation='h', showlegend=False, marker_color='blue'))
#
# fig.update_layout(
#     xaxis_title="Feature Ranking",
#     yaxis_title="Feature Names",
#     # boxmode='group',  # group together boxes of the different traces for each value of x
#     # autosize=False,
#     width=800,
#     height=1200,
#
# )
# plotly.offline.plot(fig, filename='feature_ranking.html')


# -------------------------------------
# PPM plot median and all samples
# --------------------------
# from pathlib import Path
# import pandas as pd
# import pickle
# import numpy as np
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly
#
# BASE_PATH = Path("/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments")
#
#
# def load_pickle_file(file_path):
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)
#
#
# def load_csv_file(file_path):
#     return pd.read_csv(file_path)
#
#
# def get_index_positions(data):
#     index_positions = defaultdict(list)
#     for i, sublist in enumerate(data):
#         for j, item in enumerate(sublist):
#             index_positions[item].append(j + 1)
#     return index_positions
#
#
# def create_stats_df(unique_features, index_positions):
#     df = pd.DataFrame(index=unique_features, columns=['mean', 'max', 'min', 'Q1', 'Q3', 'frequency'])
#     for feature in unique_features:
#         positions = index_positions[feature]
#         df.loc[feature, 'mean'] = np.mean(positions)
#         df.loc[feature, 'max'] = np.max(positions)
#         df.loc[feature, 'min'] = np.min(positions)
#         df.loc[feature, 'Q1'] = np.percentile(positions, 25)
#         df.loc[feature, 'Q3'] = np.percentile(positions, 75)
#         df.loc[feature, 'frequency'] = len(positions)
#     df['frequency'] = df['frequency'].astype(int)
#     return df
#
#
# def plot_ppm_with_selection(ds, pca_ds, top_features, bins_to_names, ppm_chart_title, plot_median=False):
#     def add_trace(fig, class_data, ppm_float, class_value, group, class_colour, row, col, showlegend=True,
#                   legendgroup=None):
#         fig.add_trace(go.Scatter(x=ppm_float, y=class_data.values.flatten(), mode='lines', name=group,
#                                  line=dict(color=class_colour), legendgroup=legendgroup, showlegend=showlegend),
#                       row=row, col=col)
#
#     def add_vrect(fig, min_val, max_val, row, col):
#         fig.add_vrect(x0=min_val, x1=max_val, fillcolor="green", opacity=0.25, line_width=1, row=row, col=col)
#
#     vaso_class = {0.0: "green", 1.0: "red"}
#     vaso_class1 = {0.0: "Vasoplegia Positive (plot 1)", 1.0: "Vasoplegia Negative (plot 1)"}
#     vaso_class2 = {0.0: "Vasoplegia Positive (plot 2)", 1.0: "Vasoplegia Negative (plot 2)"}
#     vaso_class3 = {0.0: "Vasoplegia Positive (plot 3)", 1.0: "Vasoplegia Negative (plot 3)"}
#
#     scaler = StandardScaler()
#     data = ds[ds.columns[2:]].to_numpy()
#     data_scaled = scaler.fit_transform(data)  # Fit and transform the complete dataset
#     ds_scaled = pd.DataFrame(data_scaled, index=ds.index, columns=ds.columns[2:])
#     ds_scaled = pd.concat([ds[ds.columns[:2]], ds_scaled], axis=1)
#
#     ppm_float = ds[ds.columns[2:]].columns.astype(float)
#     fig = make_subplots(rows=3, cols=1, shared_xaxes=False)
#
#     median_prefix = ""
#     if plot_median:
#         median_prefix = "Median "
#         unique_classes = ds["Vasoplegia"].unique()
#         for class_value in unique_classes:
#             class_data = ds[ds["Vasoplegia"] == class_value][ds.columns[2:]]
#             median_data = class_data.median()  # Compute median
#             add_trace(fig, median_data, ppm_float, class_value, vaso_class1[class_value], vaso_class[class_value], 1, 1,
#                       showlegend=True, legendgroup=vaso_class1[class_value])  # No scaling for subplot 1
#
#             class_data_scaled = ds_scaled[ds_scaled["Vasoplegia"] == class_value][ds_scaled.columns[2:]]
#             median_data_scaled = class_data_scaled.median()  # Compute median of scaled data
#             add_trace(fig, median_data_scaled, ppm_float, class_value, vaso_class2[class_value],
#                       vaso_class[class_value], 2, 1, showlegend=True,
#                       legendgroup=vaso_class2[class_value])  # Scaling for subplot 2
#
#             class_data = pca_ds[pca_ds["Vasoplegia"] == class_value][pca_ds.columns[2:]]
#             median_pca_data = class_data.median()  # Compute median of PCA data
#             add_trace(fig, median_pca_data, list(range(len(pca_ds.columns[2:]))), class_value, vaso_class3[class_value],
#                       vaso_class[class_value], 3, 1, showlegend=True,
#                       legendgroup=vaso_class3[class_value])  # No scaling for subplot 3
#     else:
#         unique_classes = ds["Vasoplegia"].unique()
#         for class_value in unique_classes:
#             group_data = ds[ds["Vasoplegia"] == class_value]
#             group_data_scaled = ds_scaled[ds_scaled["Vasoplegia"] == class_value]
#             group_data_pca = pca_ds[pca_ds["Vasoplegia"] == class_value]
#
#             for i, (row, row_scaled, row_pca) in enumerate(
#                     zip(group_data.iterrows(), group_data_scaled.iterrows(), group_data_pca.iterrows())):
#                 show_legend = i == 0  # Show legend only for the first sample in each class
#
#                 add_trace(fig, pd.DataFrame(row[1][2:]).T, ppm_float, class_value, vaso_class1[class_value],
#                           vaso_class[class_value], 1, 1, showlegend=show_legend, legendgroup=vaso_class1[class_value])
#                 add_trace(fig, pd.DataFrame(row_scaled[1][2:]).T, ppm_float, class_value, vaso_class2[class_value],
#                           vaso_class[class_value], 2, 1, showlegend=show_legend, legendgroup=vaso_class2[class_value])
#                 add_trace(fig, pd.DataFrame(row_pca[1][2:]).T, list(range(len(pca_ds.columns[2:]))), class_value,
#                           vaso_class3[class_value], vaso_class[class_value], 3, 1, showlegend=show_legend,
#                           legendgroup=vaso_class3[class_value])
#
#     for top_fs_index in list(top_features["index"]):
#         feature_names = bins_to_names[top_fs_index]
#         feature_ppm_values = [float(feature) for feature in feature_names]
#         ppm_min = min(feature_ppm_values)
#         ppm_max = max(feature_ppm_values)
#         add_vrect(fig, ppm_min, ppm_max, 1, 1)
#         add_vrect(fig, ppm_min, ppm_max, 2, 1)
#         add_vrect(fig, top_fs_index - 1, top_fs_index + 1, 3, 1)
#
#     fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=1, col=1)
#     fig.update_yaxes(title_text=median_prefix + 'Signal Intensity', row=1, col=1)
#     fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=2, col=1)
#     fig.update_yaxes(title_text=median_prefix + 'Scaled Signal Intensity', row=2, col=1)
#     fig.update_xaxes(title_text=median_prefix + 'PCA Binned Features', row=3, col=1)
#     fig.update_yaxes(title_text='PC1 Values', row=3, col=1)
#     fig.update_layout(xaxis3=dict(range=[0, len(pca_ds.columns[2:]) - 1]), title=ppm_chart_title)
#     plotly.offline.plot(fig, filename=median_prefix + 'ppm_chart.html')
#
#
# bins_to_names = load_pickle_file(
#     BASE_PATH / 'datasets/vasoplegia/cpmg_denoised__pca_reduction__bin005_overlap0025.bins_to_names')
# original_data_df = load_csv_file(BASE_PATH / 'datasets/vasoplegia/cpmg_denoised.csv')
# cpmg_pca_df = load_csv_file(BASE_PATH / 'datasets/vasoplegia/cpmg__pca_reduction__bin005_overlap0025.csv')
# ordered_ranking = load_pickle_file(
#     BASE_PATH / 'RFE_seq_attention/10_FOLDS_10_REPS/2023-05-26_20:00/cpmg_denoised__pca_reduction__bin005_overlap0025/epoch1500/auto_scaler/features_in_selection_order_path.backup')
#
# data = list(ordered_ranking)
# unique_features = list(set(item for sublist in data for item in sublist))
# index_positions = get_index_positions(data)
# df = create_stats_df(unique_features, index_positions)
# top_features = df.nlargest(50, 'frequency')
# top_features = top_features.sort_values(by='mean').reset_index()
# plot_ppm_with_selection(original_data_df, cpmg_pca_df, top_features, bins_to_names, "", plot_median=False)

