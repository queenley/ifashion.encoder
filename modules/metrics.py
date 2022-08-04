import tensorflow as tf
from tensorflow.keras import backend as K


class MulticlassPrecision(tf.keras.metrics.Metric):

    def __init__(self,
                 num_classes,
                 average=None,
                 threshold=None,
                 name='precision',
                 dtype=None,
                 **kwargs):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [None, 'micro', 'macro', 'weighted']")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError(
                    "The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(name,
                                   shape=self.init_shape,
                                   initializer="zeros",
                                   dtype=self.dtype)

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold,
                                    tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(
            _weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight))
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight))
        self.weights_intermediate.assign_add(
            _weighted_sum(y_true, sample_weight))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives)

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate,
                tf.reduce_sum(self.weights_intermediate))
            precision = tf.reduce_sum(precision * weights)

        elif self.average is not None:  # [micro, macro]
            precision = tf.reduce_mean(precision)

        return precision

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = tf.zeros(self.init_shape, dtype=self.dtype)
        K.batch_set_value([(v, reset_value) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()


class MulticlassRecall(tf.keras.metrics.Metric):

    def __init__(self,
                 num_classes,
                 average=None,
                 threshold=None,
                 name='recall',
                 dtype=None,
                 **kwargs):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [None, 'micro', 'macro', 'weighted']")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError(
                    "The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(name,
                                   shape=self.init_shape,
                                   initializer="zeros",
                                   dtype=self.dtype)

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold,
                                    tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(
            _weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight))
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight))
        self.weights_intermediate.assign_add(
            _weighted_sum(y_true, sample_weight))

    def result(self):
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate,
                tf.reduce_sum(self.weights_intermediate))
            recall = tf.reduce_sum(recall * weights)

        elif self.average is not None:  # [micro, macro]
            recall = tf.reduce_mean(recall)

        return recall

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = tf.zeros(self.init_shape, dtype=self.dtype)
        K.batch_set_value([(v, reset_value) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()
