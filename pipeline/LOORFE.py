import tensorflow as tf
import pandas as pd
from collections import defaultdict
import numpy as np
import os
import copy
from plots.live_learning import PlotLearning
from tensorflow import keras


class LeaveOneOutRecursiveFeatureElimination:
    """
        Implements an unbiased Leave-One-Out Recursive Feature Elimination (LOO-RFE) approach for
        feature selection in machine learning models. This class leverages cross-validation (CV)
        splits to generate feature rankings that exclude the influence of the test subject, thereby
        providing an unbiased evaluation of feature importance.

        The LOO-RFE method iteratively eliminates features based on their impact on model performance,
        utilizing a unique approach to ensure that the evaluation of each feature's importance is not
        biased by the inclusion of the test subject's data in the training set. This is achieved by
        constructing feature rankings from subsets of data where the test subject was excluded, using
        these unbiased rankings to identify the most relevant features for model prediction.

        Parameters:
        - data_handler: An object responsible for managing data operations such as scaling and batching,
                        ensuring that data is appropriately preprocessed for model training and evaluation.
        - model_eval_class: A class that provides methods for model compilation, fitting, and evaluation,
                            designed to work with variable numbers of features and target classes.
        - num_classes (int): The number of target classes in the dataset, defaulting to 2 for binary classification.
        - seed (int, optional): A random seed to ensure reproducibility of results across runs.

        Methods:
        - get_keys_where_value_exists: Retrieves keys from a list of dictionaries where a specified value is present,
                                       aiding in the identification of CV splits relevant to each test subject.
        - get_top_features: Extracts top features based on their occurrence and average position in ordered rankings,
                            utilizing an unbiased ranking list compiled from CV splits excluding the test subject.
        - evaluate: Conducts the LOO-RFE evaluation, training models on subsets of features and assessing their performance
                    in an unbiased manner, guided by the principle of excluding the test subject's data from feature ranking
                    generation.

        This class provides a robust framework for feature selection by prioritizing the most impactful features
        for model performance while ensuring the evaluation process is unbiased by the test data. The approach
        enhances the generalizability and reliability of the selected features, making it a valuable tool for
        machine learning tasks that require careful feature selection and validation.
        """
    def __init__(self, data_handler, model_eval_class, num_classes=2, seed=None):
        """
        Initializes the LeaveOneOutRecursiveFeatureElimination class with specified parameters.
        """
        self.data_handler = data_handler
        self.model_eval_class = model_eval_class
        self.num_classes = num_classes
        self.seed = seed

    def get_keys_where_value_exists(self, dict_list, X):
        """
        Finds keys in a list of dictionaries where a specified value exists.

        Parameters:
        - dict_list (list of dict): The list of dictionaries to search.
        - X (int): The value to search for within the dictionary values.

        Returns:
        - keys (list of str): A list of keys where the specified value is present among the values.
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

    def get_top_features(self, ordered_rankings, ascending=False, n_features=None):
        """
        Identifies top features based on their occurrence and average position in ordered rankings.

        Parameters:
        - ordered_rankings (list of lists): A list where each sublist represents feature rankings in an iteration.
        - ascending (bool): Determines the sorting order of mean positions. Defaults to False.
        - n_features (int, optional): Specifies the number of top features to select. Defaults to selecting all.

        Returns:
        - top_features (DataFrame): A DataFrame containing statistics of top features including frequency and position metrics.
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

    def evaluate(self, data, train_index, test_index, batch_size, num_selected_features, sample_num, subs_in_val_per_sample, selected_feature_indices, scaler_obj, save_path):
        """
        Performs the LOO-RFE evaluation process, training models on subsets of features and assessing their performance.

        Parameters:
        - data: The dataset containing features and labels.
        - train_index: Indices for the training data.
        - test_index: Indices for the test data.
        - batch_size: The size of batches for training and testing.
        - num_selected_features: The number of features to select in the current iteration.
        - sample_num: Identifier for the current sample or iteration.
        - subs_in_val_per_sample: Subsets involved in validation for each sample.
        - selected_feature_indices: Indices of features selected in the current iteration.
        - scaler_obj: An instance of a scaler for data normalization.
        - save_path: Path where evaluation results and plots are saved.

        Returns:
        - A DataFrame containing evaluation results across different feature subsets.
        """

        tf.random.set_seed(self.seed)
        Xs, ys = data

        # Identify unbiased rankings for the test index
        indexes = self.get_keys_where_value_exists(subs_in_val_per_sample, test_index)
        unbiased_rankings = [selected_feature_indices[i] for i in indexes]

        # Merge the rankings into one accordingly to selection frequency and mean position
        top_ranking = self.get_top_features(ordered_rankings=unbiased_rankings, ascending=True, n_features=num_selected_features)

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

        # Transform datasets
        ds_train, ds_test = self.data_handler.create_batched_datasets(x_train_scaled, y_train, x_test_scaled, y_test, batch_size)

        # Define the scoring metrics
        f1 = tf.keras.metrics.F1Score(average="macro", threshold=None, name='f1_score', dtype=None)
        auc = tf.keras.metrics.AUC(name="auc_score", dtype=None)

        features_in_selection_order = top_ranking.index

        loo_rfe_res = []

        while len(features_in_selection_order) > 1:
            ds_train_selected = self.data_handler.reduce_data_to_specific_features(ds_train, list(features_in_selection_order))
            ds_test_selected = self.data_handler.reduce_data_to_specific_features(ds_test, list(features_in_selection_order))
            # Create and train the model
            model = self.model_eval_class(len(features_in_selection_order), [f1, auc], self.num_classes)
            # model = MLPEval(len(features_in_selection_order), [f1, auc], self.num_classes)

            loo_rfe_eval_train_plots_save_path = save_path + "/loo_rfe_eval_training_progress_plots/{}_features".format(len(features_in_selection_order))
            loo_rfe_title = "LOO RFE - {} features".format(len(features_in_selection_order))
            os.makedirs(loo_rfe_eval_train_plots_save_path, exist_ok=True)
            save_eval_plot_as = loo_rfe_eval_train_plots_save_path + "/loo_rfe_eval_train_sample-{}.png".format(sample_num)

            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

            model.fit(
                ds_train_selected,
                epochs=50,
                validation_data=ds_test_selected,
                callbacks=[early_stop, PlotLearning(loo_rfe_title, save_eval_plot_as)]
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

