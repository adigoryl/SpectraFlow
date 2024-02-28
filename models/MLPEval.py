import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

class MLPEval(models.Sequential):
    """Shallow MLP model for evaluation."""

    def __init__(
            self,
            n_inputs,
            metrics,
            num_classes,
            units=9,
            activation='relu',
            l1_reg=0.005,
            l2_reg=0.0005,
            dropout_rate=0.4,
            learning_rate=0.0005
    ):
        """
        Initialize the shallow MLP model.

        Parameters:
            n_inputs (int): Number of input features.
            metrics (list): List of metrics to be used for model evaluation.
            num_classes (int): Number of output classes.
            units (int): Number of units in the hidden layer.
            activation (str): Activation function for the hidden layer.
            l1_reg (float): L1 regularization factor.
            l2_reg (float): L2 regularization factor.
            dropout_rate (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.add(layers.Input(shape=(n_inputs,)))
        self.add(layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
        self.add(layers.Dropout(dropout_rate))
        self.add(layers.Dense(num_classes, activation='sigmoid'))

        self.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=metrics
        )

# Example usage:
# model = MLPEval(n_inputs=100, metrics=['accuracy'], num_classes=2)
