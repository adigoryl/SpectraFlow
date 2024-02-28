import tensorflow as tf
import pandas as pd
import pickle
import os
import copy

from sklearn.model_selection import RepeatedStratifiedKFold


class DataHandler:
    def __init__(self, num_classes=2, root=None, dataset_path=None, seed=None):
        self.num_classes = num_classes
        self.root = root
        self.dataset_path = dataset_path
        self.seed = seed

    def load_dataset(self, datasets_to_combine):
        comb = pd.concat([pd.read_csv(f"{self.root}/{self.dataset_path}/{name}.csv") for name in datasets_to_combine],axis=1).dropna()

        comb = comb.loc[:, ~comb.columns.duplicated()]
        comb_name = '+'.join(datasets_to_combine)

        class_key = "Vasoplegia"
        Xs = comb.drop(columns=[class_key, "Subject ID"])
        ys = comb[class_key].astype("int8")

        return comb_name, Xs, ys

    def prepare_data_folds(self, datasets_to_combine, n_cv_folds, n_cv_repetitions, scaler_obj):
        comb_name, Xs, ys = self.load_dataset(datasets_to_combine)

        cv_obj = RepeatedStratifiedKFold(n_splits=n_cv_folds, n_repeats=n_cv_repetitions, random_state=self.seed)
        list_of_data_dicts = []
        subs_in_val_per_sample = []

        for i, (train_index, val_index) in enumerate(cv_obj.split(Xs, ys)):
            x_train, y_train = Xs.iloc[train_index], ys.iloc[train_index]
            x_val, y_val = Xs.iloc[val_index], ys.iloc[val_index]

            if scaler_obj is not None:
                scaler = copy.deepcopy(scaler_obj)
                x_train_scaled = scaler.fit_transform(x_train)
                x_val_scaled = scaler.transform(x_val)
            else:
                x_train_scaled = copy.deepcopy(x_train)
                x_val_scaled = copy.deepcopy(x_val)

            data_dict = {
                "x_train": x_train,
                "y_train": y_train,
                "x_val": x_val,
                "y_val": y_val,
                "x_train_scaled": x_train_scaled,
                "x_val_scaled": x_val_scaled,
                "num_features": x_train.shape[1],
                "num_classes": self.num_classes,
                "is_classification": True,
            }
            list_of_data_dicts.append(data_dict)
            subs_in_val_per_sample.append({i: val_index.tolist()})

        return comb_name, list_of_data_dicts, subs_in_val_per_sample

    @staticmethod
    def reduce_data_to_specific_features(batched_data, selected_indices_list):
        selected_indices_tensor = tf.constant(selected_indices_list, dtype=tf.int32)

        def select_features(batch_x, batch_y):
            return tf.gather(batch_x, selected_indices_tensor, axis=1), batch_y

        return batched_data.map(select_features)

    def transform(self, x, y):
        x = tf.cast(x, dtype=tf.float32)
        return x, tf.one_hot(y, self.num_classes)

    @staticmethod
    def pickle_dump(file_path, obj):
        with open(file_path, 'wb+') as filehandler:
            pickle.dump(obj, filehandler)

    @staticmethod
    def calculate_metrics(df):
        from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
        import pandas as pd

        y_true = df['true_label']
        y_pred = df['y_pred']
        y_score = df['y_pos_pred_probability']

        return pd.Series({
            'f1': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_score),
            'positive_precision': precision_score(y_true, y_pred, pos_label=1),
            'positive_recall': recall_score(y_true, y_pred, pos_label=1),
            'negative_precision': precision_score(y_true, y_pred, pos_label=0),
            'negative_recall': recall_score(y_true, y_pred, pos_label=0)
        })

    def create_batched_datasets(self, x_train_scaled, y_train, x_val_scaled, y_val, batch_size):
        buffer_size = x_train_scaled.shape[0]

        ds_train = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train))
        ds_train = ds_train.map(self.transform).shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)

        ds_val = tf.data.Dataset.from_tensor_slices((x_val_scaled, y_val))
        ds_val = ds_val.map(self.transform).batch(batch_size, drop_remainder=False)

        return ds_train, ds_val

    @staticmethod
    def ensure_directories(paths):
        for path in paths:
            os.makedirs(path, exist_ok=True)
