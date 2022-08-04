import atexit
import os

import modules.dataset as dataset
import tensorflow as tf
import tensorflow_addons as tfa
from absl import app, flags
from absl.flags import FLAGS
from modules.metrics import MulticlassPrecision, MulticlassRecall
from modules.models import ArcFaceModel
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import TensorBoard
from utils.utils import load_yaml

flags.DEFINE_string('cfg_path', './configs/arc_res50_lookbook.yaml',
                    'config file path')
flags.DEFINE_string('name', 'lookbook_m01_s16_64_pants', 'name of the model')


def main(_):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    cfg = load_yaml(FLAGS.cfg_path)

    with mirrored_strategy.scope():
        model = ArcFaceModel(size=cfg['input_size'],
                             backbone_type=cfg['backbone_type'],
                             num_classes=cfg['num_classes'],
                             head_type=cfg['head_type'],
                             margin=cfg['margin'],
                             easy_margin=cfg['easy_margin'],
                             logist_scale=cfg['logist_scale'],
                             embd_shape=cfg['embd_shape'],
                             w_decay=cfg['w_decay'],
                             training=True)
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
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['base_lr'],
                                            momentum=0.9,
                                            nesterov=True)
    model.summary()

    test_dataset = dataset.load_tfrecord_dataset(
        tfrecord_name=cfg['test_dataset'],
        batch_size=cfg['batch_size'],
        num_classes=cfg['num_classes'],
        size=cfg['input_size'],
        is_train=False,
        shuffle=False,
        repeat=False)

    ckpt_dir_name = cfg['sub_name'] + '_' + FLAGS.name
    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + ckpt_dir_name)
    if ckpt_path is not None:
        print("[*] Load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path).expect_partial()
    else:
        raise ValueError("[!] No ckpt found")

    tb_callback = TensorBoard(log_dir='./logs/' + ckpt_dir_name)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=metrics)
    model.evaluate(test_dataset, callbacks=[tb_callback])

    atexit.register(mirrored_strategy._extended._collective_ops._pool.close)


if __name__ == '__main__':
    app.run(main)
