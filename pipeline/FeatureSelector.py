import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from plots.live_learning import PlotLearning

class FeatureSelector:
    def __init__(self, data_handler, model_select_class, model_eval_class, fs_args, mlp_args, num_epochs_select, num_epochs_evaluator, num_selected_features, seed):
        self.data_handler = data_handler
        self.model_select_class = model_select_class
        self.model_eval_class = model_eval_class
        self.fs_args = fs_args
        self.mlp_args = mlp_args
        self.num_epochs_select = num_epochs_select
        self.num_epochs_evaluator = num_epochs_evaluator
        self.num_selected_features = num_selected_features
        self.seed = seed

    def compile_model(self, model, metrics):
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

    def feature_selection_and_eval(self, data, batch_size, sample_num, save_path):
        tf.random.set_seed(self.seed)
        self.data_handler.ensure_directories([save_path + "/fs_training_n_val_progress", save_path + "/evaluator_training_n_val_progress"])

        ds_train, ds_val = self.data_handler.create_batched_datasets(data["x_train_scaled"], data["y_train"], data["x_val_scaled"], data["y_val"], batch_size)

        # Setup common metrics
        metrics = [tf.keras.metrics.F1Score(name='f1_score', average='macro'), tf.keras.metrics.AUC(name='auc_score'),
                   tf.keras.metrics.Precision(name='positive_precision', class_id=1),
                   tf.keras.metrics.Recall(name='positive_recall', class_id=1),
                   tf.keras.metrics.Precision(name='negative_precision', class_id=0),
                   tf.keras.metrics.Recall(name='negative_recall', class_id=0)]

        # Set the missing hyperparameters
        self.mlp_args["num_classes"] = data["num_classes"]
        self.mlp_args["is_classification"] = data["is_classification"]
        self.fs_args["num_inputs"] = data["num_features"]
        self.fs_args["num_train_steps"] = self.num_epochs_select * len(ds_train)

        # Initialize and compile the selection model
        mlp_select = self.model_select_class(**{**self.mlp_args, **self.fs_args})
        self.compile_model(mlp_select, metrics)

        train_plot_title = "FS TRAINING -> sample-{} :: select={}, epoch={}, batch={}".format(sample_num, self.num_selected_features, self.num_epochs_select, batch_size)
        save_train_plot_as = save_path + "/fs_training_n_val_progress/fs_sample-{}.png".format(sample_num)

        mlp_select.fit(
            ds_train,
            validation_data=ds_val,
            epochs=self.num_epochs_select,
            verbose=2,
            callbacks=[PlotLearning(train_plot_title, save_train_plot_as)]
        )

        selected_features = mlp_select.seqatt.selected_features
        _, selected_indices = tf.math.top_k(selected_features, k=self.num_selected_features)
        selected_indices = selected_indices.numpy()

        ds_train_selected = self.data_handler.reduce_data_to_specific_features(ds_train, list(selected_indices))
        ds_val_selected = self.data_handler.reduce_data_to_specific_features(ds_val, list(selected_indices))

        # Initialize, compile, and train the evaluation model
        model = self.model_eval_class(self.num_selected_features, metrics, data["num_classes"])
        self.compile_model(model, metrics)

        eval_plot_title = "Validation -> sample-{} :: select={}, epoch={}, batch={}".format(sample_num, self.num_selected_features, self.num_epochs_select, batch_size)
        eval_train_plot_path = save_path + "/evaluator_training_n_val_progress/evaluator_sample-{}.png".format(sample_num)
        model.fit(
            ds_train_selected,
            validation_data=ds_val_selected,
            epochs=self.num_epochs_evaluator,
            verbose=2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5), PlotLearning(eval_plot_title, eval_train_plot_path)]
        )

        # Evaluate the model and prepare the return values
        results = model.evaluate(ds_val_selected, return_dict=True)
        evaluation_results = {
            "val_f1_shallow": results["f1_score"],
            "val_loss_shallow": results["loss"],
            "val_auc_shallow": results["auc_score"],
            "val_positive_precision_shallow": results["positive_precision"],
            "val_positive_recall_shallow": results["positive_recall"],
            "val_negative_precision_shallow": results["negative_precision"],
            "val_negative_recall_shallow": results["negative_recall"]
        }

        features_in_selection_order = mlp_select.get_feature_selection_order().numpy()

        return list(selected_indices), evaluation_results, features_in_selection_order
