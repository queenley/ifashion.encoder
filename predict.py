import os

import numpy as np
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from sklearn import metrics

from utils.utils import load_yaml

flags.DEFINE_string('cfg_path', './configs/arc_res50_lookbook.yaml',
                    'config file path')
flags.DEFINE_string('input', './input.jpg', 'input image path')
flags.DEFINE_string('saved_model', './encoder.h5', 'weight path')


def main(_):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    cfg = load_yaml(FLAGS.cfg_path)

    # Load encoder.h5
    model = tf.keras.models.load_model(FLAGS.saved_model)
    model.summary()

    # iid = np.load('iid.npy')
    # embeddings = np.load('embeddings.npy')
    # dummpy_label = np.zeros((1, cfg['num_classes']))

    input_image = tf.keras.preprocessing.image.load_img(FLAGS.input)
    input_image = tf.keras.preprocessing.image.img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)
    input_image /= 255.0

    embedding = model.predict((input_image))
    print(embedding)
    # # Cosine distance
    # dist = metrics.pairwise.cosine_distances(embeddings, embedding)
    # dist = np.squeeze(dist)

    # # Min 10 index of the distance
    # idx = np.argsort(dist)[:10]

    # print("[*] Top 10 similar images:")
    # for i in idx:
    #     print("IID: {}, distance: {}".format(iid[i], dist[i]))


if __name__ == '__main__':
    app.run(main)
