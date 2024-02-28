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

"""Sequential Attention for Feature Selection.

This module implements a Sequential Attention mechanism for feature selection as described in
https://arxiv.org/abs/2209.14881. It progressively selects features based on their importance
calculated through a trainable attention mechanism, optimizing for both feature relevance and
compactness in the selected feature set.
"""

import tensorflow as tf


class SequentialAttention(tf.Module):
    """Implements a Sequential Attention mechanism for feature selection.

    This class defines a module that applies a trainable attention mechanism to sequentially select
    a subset of features from a larger set based on their relevance. The selection process is
    controlled by a percentage of the training process completed, allowing for dynamic adjustment
    of feature importance over time.
    """

    def __init__(
            self,
            num_candidates,
            num_candidates_to_select,
            num_candidates_to_select_per_step=1,
            start_percentage=0.1,
            stop_percentage=1.0,
            name='sequential_attention',
            reset_weights=True,
            **kwargs,
    ):
        """Initializes the SequentialAttention module.

        Args:
          num_candidates: Total number of candidate features.
          num_candidates_to_select: Number of features to select.
          num_candidates_to_select_per_step: Number of features to select at each step.
          start_percentage: Training percentage at which to start feature selection.
          stop_percentage: Training percentage at which to stop feature selection.
          name: Name of the module.
          reset_weights: Whether to reset attention weights after each selection.
          **kwargs: Additional keyword arguments.
        """
        super(SequentialAttention, self).__init__(name=name, **kwargs)

        assert num_candidates_to_select % num_candidates_to_select_per_step == 0, (
            'num_candidates_to_select must be a multiple of '
            'num_candidates_to_select_per_step.'
        )

        with self.name_scope:
            self._num_candidates = num_candidates
            self._num_candidates_to_select_per_step = num_candidates_to_select_per_step
            self._num_steps = num_candidates_to_select // num_candidates_to_select_per_step
            self._start_percentage = start_percentage
            self._stop_percentage = stop_percentage
            self._reset_weights = reset_weights

            init_attention_weights = tf.random.normal(
                shape=[num_candidates], stddev=0.00001, dtype=tf.float32
            )
            self._attention_weights = tf.Variable(
                initial_value=lambda: init_attention_weights,
                dtype=tf.float32,
                name='attention_weights',
            )

            self.selected_features = tf.Variable(
                tf.zeros(shape=[num_candidates], dtype=tf.float32),
                trainable=False,
                name='selected_features',
            )

        # --------------  Not a part of original code ------------
        self.selected_feature_order = tf.Variable(
            tf.zeros(shape=[self._num_steps], dtype=tf.int32),
            trainable=False,
            name='selected_feature_order',
        )
        # -------------------------------------------------------

    @tf.Module.with_name_scope
    def __call__(self, training_percentage):
        """Calculates and updates attention weights for feature selection.

        This method computes the attention weights for each feature, selects features based on
        the computed weights, and optionally resets weights after selection.

        Args:
          training_percentage: A float representing the current training progress as a percentage.

        Returns:
          A tensor representing the updated vector of attention weights for all candidates.
        """
        percentage = (training_percentage - self._start_percentage) / (self._stop_percentage - self._start_percentage)
        curr_index = tf.cast(tf.math.floor(percentage * self._num_steps), dtype=tf.float32)
        curr_index = tf.math.minimum(curr_index, self._num_steps - 1.0)

        should_train = tf.less(curr_index, 0.0)
        num_selected = tf.math.reduce_sum(self.selected_features)
        should_select = tf.greater_equal(curr_index, num_selected)

        _, new_indices = tf.math.top_k(
            self._softmax_with_mask(self._attention_weights, 1.0 - self.selected_features),
            k=self._num_candidates_to_select_per_step,
        )

        new_indices = self._k_hot_mask(new_indices, self._num_candidates)
        new_indices = tf.cond(
            should_select,
            lambda: new_indices,
            lambda: tf.zeros(self._num_candidates),
        )
        select_op = self.selected_features.assign_add(new_indices)

        # --------------  Not a part of original code ------------
        track_index_op = tf.cond(
            tf.less(num_selected, self._num_steps),
            lambda: self.selected_feature_order.scatter_nd_update(
                indices=tf.reshape(tf.cast(num_selected, tf.int32), [1, 1]),
                updates=tf.reshape(tf.cast(tf.argmax(new_indices), tf.int32), [1])
            ),
            lambda: self.selected_feature_order
        )
        # -------------------------------------------------------

        init_attention_weights = tf.random.normal(
            shape=[self._num_candidates], stddev=0.00001, dtype=tf.float32
        )
        should_reset = tf.logical_and(should_select, self._reset_weights)
        new_weights = tf.cond(
            should_reset,
            lambda: init_attention_weights,
            lambda: self._attention_weights,
        )
        reset_op = self._attention_weights.assign(new_weights)

        with tf.control_dependencies([select_op, reset_op, track_index_op]):
            candidates = 1.0 - self.selected_features
            softmax = self._softmax_with_mask(self._attention_weights, candidates)
            return tf.cond(
                should_train,
                lambda: tf.ones(self._num_candidates),
                lambda: softmax + self.selected_features,
            )

    @tf.Module.with_name_scope
    def _k_hot_mask(self, indices, depth, dtype=tf.float32):
        """Generates a k-hot mask for the given indices.

        Args:
          indices: A tensor of indices.
          depth: The depth of the one-hot encoding.
          dtype: The data type of the resulting tensor.

        Returns:
          A k-hot encoded tensor.
        """
        return tf.math.reduce_sum(tf.one_hot(indices, depth, dtype=dtype), 0)

    @tf.Module.with_name_scope
    def _softmax_with_mask(self, logits, mask):
        """Applies softmax function to logits with a mask.

        Args:
          logits: A tensor containing logits.
          mask: A tensor containing a mask to apply to the logits.

        Returns:
          A tensor with softmax applied to the masked logits.
        """
        shifted_logits = logits - tf.math.reduce_max(logits)
        exp_shifted_logits = tf.math.exp(shifted_logits)
        masked_exp_shifted_logits = tf.multiply(exp_shifted_logits, mask)
        return tf.math.divide_no_nan(
            masked_exp_shifted_logits, tf.math.reduce_sum(masked_exp_shifted_logits)
        )

    def get_feature_selection_order(self):
        """Get the order in which features were selected."""
        return self.selected_feature_order
