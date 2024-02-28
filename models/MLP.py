import tensorflow as tf


class MLPModel2(tf.keras.Model):
    """A customizable Multi-Layer Perceptron (MLP) model for deep learning.

    This class allows for the creation of a flexible MLP architecture for either classification
    or regression tasks. It supports features such as batch normalization, dropout for regularization,
    and dynamic learning rate decay.

    Attributes:
        batch_norm_layer (tf.keras.layers.BatchNormalization): Optional batch normalization layer.
        batch_norm (bool): Flag indicating whether batch normalization is to be used.
        mlp_model (tf.keras.Sequential): Sequential model representing the MLP layers.
        mlp_predictor (tf.keras.layers.Dense): Final dense layer for prediction.
        optimizer (tf.keras.optimizers.Adam): Optimizer with a decaying learning rate.
    """

    def __init__(
            self,
            layer_sequence=[1],
            is_classification=True,
            num_classes=None,
            learning_rate=0.001,
            decay_steps=500,
            decay_rate=0.8,
            alpha=0,
            batch_norm=True,
            dropout_rate=0.5,
    ):
        """Initializes the MLPModel2 instance.

        Parameters:
            layer_sequence (list, optional): List of integers defining the number of units in each layer.
            is_classification (bool): True if the model is used for classification, False for regression.
            num_classes (int, optional): Number of classes for classification. Required if is_classification=True.
            learning_rate (float): Initial learning rate for the optimizer.
            decay_steps (int): Number of steps before applying one decay step.
            decay_rate (float): Decay rate for the learning rate.
            alpha (float): Negative slope coefficient for LeakyReLU.
            batch_norm (bool): Whether to include a batch normalization layer.
            dropout_rate (float): Fraction of the input units to drop.
        """
        super().__init__()

        # Initialize batch normalization layer if enabled.
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

        # Create the MLP layers based on the provided layer sequence.
        mlp_sequence = []
        for dim in layer_sequence:
            mlp_sequence.append(tf.keras.layers.Dense(dim, activation=tf.keras.layers.LeakyReLU(alpha=alpha)))
            mlp_sequence.append(tf.keras.layers.Dropout(dropout_rate))

        self.mlp_model = tf.keras.Sequential(mlp_sequence)

        # Configure the output layer for classification or regression.
        if is_classification:
            self.mlp_predictor = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")
        else:
            self.mlp_predictor = tf.keras.layers.Dense(1, dtype="float32")

        # Set up the optimizer with exponential decay for the learning rate.
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def call(self, inputs, training=False):
        """Performs a forward pass through the model.

        Parameters:
            inputs (tensor): Input data.
            training (bool): Whether the call is during training (affects dropout and batch normalization).

        Returns:
            The model's prediction output.
        """
        if self.batch_norm:
            inputs = self.batch_norm_layer(inputs, training=training)

        representation = self.mlp_model(inputs, training=training)
        prediction = self.mlp_predictor(representation)
        return prediction
