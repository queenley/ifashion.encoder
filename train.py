import atexit
import os

import tensorflow as tf
import tensorflow_addons as tfa
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import modules.dataset as dataset
from modules.metrics import MulticlassPrecision, MulticlassRecall
from modules.models import ArcFaceModel
from utils.utils import get_ckpt_inf, load_yaml

flags.DEFINE_string('cfg_path', './configs/arc_res50_lookbook.yaml',
                    'config file path')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager'],
                  'fit: model.fit, eager: custom GradientTape')
flags.DEFINE_string('name', 'lookbook', 'name of the model')


class ModelOutput(tf.keras.metrics.Metric):
    ''' Class wrapper for a metric that stores the output passed to it '''

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(tf.cast(y_true, tf.int64))

        self.y_true = y_true
        self.y_pred = y_pred
        return tf.constant(True)

    def result(self):
        tf.py_function(write_string, [self.y_true, self.y_pred], tf.bool)
        return True


def write_string(y_true, y_pred):
    with open('./results.txt', 'a') as f:
        for yt, yp in zip(y_true, y_pred):
            f.write(str(yt.numpy()) + ' ' + str(yp.numpy()) + '\n')
    return True


class ModelOutputCallback(tf.keras.callbacks.Callback):

    def __init__(self, model_outputs):
        tf.keras.callbacks.Callback.__init__(self)
        self.model_outputs = model_outputs

    def on_epoch_end(self, epoch, logs=None):
        self.model_outputs.save_to_file(['./y_true.txt', './y_pred.txt'])


def create_model(cfg, training=True):
    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         margin=cfg['margin'],
                         easy_margin=cfg['easy_margin'],
                         logist_scale=cfg['logist_scale'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=training)
    return model


def main(_):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)

    learning_rate = tf.constant(cfg['base_lr'])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    with mirrored_strategy.scope():
        model = create_model(cfg, training=True)
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            MulticlassPrecision(name='precision',
                                num_classes=cfg["num_classes"],
                                average='macro'),
            MulticlassRecall(name='recall',
                             num_classes=cfg["num_classes"],
                             average='macro'),
            tfa.metrics.F1Score(name='f1_macro',
                                num_classes=cfg["num_classes"],
                                average='macro'),
        ]
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=0.9,
                                            nesterov=True)
    model.summary()

    if cfg['train_dataset']:
        logging.info("load fashion dataset.")
        dataset_len = cfg['num_samples_train']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset.load_tfrecord_dataset(
            tfrecord_name=cfg['train_dataset'],
            batch_size=cfg['batch_size'],
            num_classes=cfg['num_classes'],
            size=cfg['input_size'])
    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1
        train_dataset = dataset.load_fake_dataset(cfg['input_size'])

    val_dataset = dataset.load_tfrecord_dataset(
        tfrecord_name=cfg['val_dataset'],
        batch_size=cfg['batch_size'],
        num_classes=cfg['num_classes'],
        size=cfg['input_size'],
        is_train=False,
        shuffle=False,
        repeat=False)

    ckpt_dir_name = cfg['sub_name'] + '_' + FLAGS.name
    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + ckpt_dir_name)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        os.makedirs('./checkpoints/' + ckpt_dir_name, exist_ok=True)
        epochs, steps = 1, 1

    mc_callback = ModelCheckpoint('checkpoints/' + ckpt_dir_name +
                                  '/e_{epoch}.ckpt',
                                  save_weights_only=True)
    tb_callback = TensorBoard(log_dir='logs/' + ckpt_dir_name, profile_batch=0)
    callbacks = [mc_callback, tb_callback]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    model.fit(train_dataset,
              epochs=cfg['epochs'],
              validation_data=val_dataset,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              initial_epoch=epochs - 1)

    infer_model = create_model(cfg, training=False)
    infer_model.load_weights(ckpt_path).expect_partial()
    infer_model.save('./encoder.h5')

    print("[*] training done!")
    atexit.register(mirrored_strategy._extended._collective_ops._pool.close)


if __name__ == '__main__':
    app.run(main)
