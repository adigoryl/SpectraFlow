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

"""
Implements a machine learning model that combines a Multi-Layer Perceptron (MLP) with Sequential Attention
for feature selection. This approach aims to enhance model performance by focusing on the most relevant input features
over a series of training steps.
"""

from models.MLP import MLPModel2  # Import base MLP model class
from models.SequentialAttention import SequentialAttention  # Import Sequential Attention mechanism
import tensorflow as tf  # Import TensorFlow for model building and operations


class SequentialAttentionModel(MLPModel2):
    """
    Defines an MLP model enhanced with a Sequential Attention mechanism for dynamic feature selection during training.

    Inherits from MLPModel2, extending its functionality with the Sequential Attention mechanism to select a subset
    of input features based on their importance to the prediction task.
    """

    def __init__(
            self,
            num_inputs,
            num_inputs_to_select,
            num_train_steps,
            num_inputs_to_select_per_step=1,
            **kwargs,
    ):
        """
    Initializes the Sequential Attention Model with specific parameters for feature selection.

    Args:
      num_inputs (int): Total number of input features.
      num_inputs_to_select (int): Number of input features to select for the model to consider.
      num_train_steps (int): Total number of training steps, used to adjust the feature selection process over time.
      num_inputs_to_select_per_step (int): Number of features to select at each selection step.
      **kwargs: Additional keyword arguments passed to the MLPModel2 initializer.
    """

        super(SequentialAttentionModel, self).__init__(**kwargs)  # Initialize the base MLP model
        self.seqatt = SequentialAttention(
            num_candidates=num_inputs,
            num_candidates_to_select=num_inputs_to_select,
            num_candidates_to_select_per_step=num_inputs_to_select_per_step,
        )  # Initialize Sequential Attention mechanism
        self.num_train_steps = num_train_steps  # Store the total number of training steps

    def call(self, inputs, training=False):
        """
        Forward pass for the model. Applies batch normalization (if enabled), Sequential Attention to input features,
        followed by the MLP layers and output prediction layer.

        Args:
          inputs (Tensor): Input features.
          training (bool): Whether the model is in training mode.

        Returns:
          Tensor: The output predictions of the model.
        """
        if self.batch_norm:
            inputs = self.batch_norm_layer(inputs, training=training)  # Apply batch normalization

        training_percentage = self.optimizer.iterations / self.num_train_steps  # Calculate training progress
        feature_weights = self.seqatt(training_percentage)  # Get feature weights from Sequential Attention
        inputs = tf.multiply(inputs, feature_weights)  # Apply feature weights to inputs
        representation = self.mlp_model(inputs)  # Process inputs through the MLP layers
        prediction = self.mlp_predictor(representation)  # Generate predictions
        return prediction

    def get_feature_selection_order(self):
        """
        Retrieves the order of feature selection determined by the Sequential Attention mechanism.

        Returns:
          List[int]: The order in which features were selected based on their importance.
        """
        return self.seqatt.get_feature_selection_order()
