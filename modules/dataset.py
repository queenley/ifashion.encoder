import tensorflow as tf


def _parse_tfrecord(is_train=True, num_classes=8726, size=256, crop=None):

    def parse_tfrecord(tfrecord):
        features = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'pid': tf.io.FixedLenFeature([], tf.int64),
            'iid': tf.io.FixedLenFeature([], tf.int64),
            'clean': tf.io.FixedLenFeature([], tf.int64)
        }
        x = tf.io.parse_single_example(tfrecord, features)
        x_train = tf.image.decode_jpeg(x['image_raw'], channels=3)
        x_train = tf.cast(x_train, tf.float32)

        y_train = tf.cast(x['pid'], tf.int32)
        y_train = _transform_targets(y_train, num_classes)

        x_train = _transform_images(is_train, size, crop)(x_train)
        x_train = tf.clip_by_value(x_train, 0.0, 1.0)

        return (x_train, y_train), y_train
        # return (x_train, y_train), y_train, x['iid'], x['clean']

    return parse_tfrecord


def _transform_images(is_train=True, size=256, crop=None):

    def transform_images(x_train):
        if is_train:
            if crop == 'random':
                x_train = tf.image.resize_with_pad(x_train, int(size * 1.5),
                                                   int(size * 1.5))
                x_train = tf.image.random_crop(x_train, [size, size, 3])
            elif crop == 'center':
                x_train = tf.image.resize_with_pad(x_train, int(size * 1.5),
                                                   int(size * 1.5))
                x_train = tf.image.central_crop(x_train,
                                                central_fraction=2 / 3)
            else:
                x_train = tf.image.resize_with_pad(x_train, size, size)
            x_train = tf.image.random_flip_left_right(x_train)
            x_train = tf.image.random_flip_up_down(x_train)
            x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
            x_train = tf.image.random_brightness(x_train, 0.1)
            x_train = tf.image.random_contrast(x_train, 0.6, 1.4)
        else:
            x_train = tf.image.resize_with_pad(x_train, size, size)
        return x_train / 255.0

    return transform_images


def _transform_targets(y_train, num_classes):
    y_train = tf.one_hot(y_train, depth=num_classes)
    return y_train


def load_tfrecord_dataset(tfrecord_name,
                          batch_size,
                          size=256,
                          shuffle=True,
                          buffer_size=10240,
                          repeat=True,
                          num_classes=8726,
                          crop=None,
                          is_train=True):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    if repeat:
        raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(_parse_tfrecord(is_train,
                                              size=size,
                                              num_classes=num_classes,
                                              crop=crop),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_fake_dataset(size):
    """load fake dataset"""
    x_train = tf.image.decode_jpeg(open('./data/BruceLee.JPG', 'rb').read(),
                                   channels=3)
    x_train = tf.expand_dims(x_train, axis=0)
    x_train = tf.image.resize(x_train, (size, size))

    labels = [0]
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
