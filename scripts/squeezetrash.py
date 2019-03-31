import tensorflow as tf
from tensorflow import keras
import collections
from datetime import datetime
import hashlib
import os.path
import random
import numpy as np
import re
import pathlib
import sys
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def load_img(img_path):
    img_raw = tf.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw)
    img.set_shape([512,384,3])
    img_final = tf.image.resize_images(img,[224,224])

    return img_final

def loadDataset(dataset):
    train_img = []
    train_label = []
    test_img = []
    test_label = []
    val_img = []
    val_label = []
    label_id = 0

    data_root = "/deepeval/dataset-resized"
    data_root = pathlib.Path(data_root)

    all_paths = list(data_root.glob('*/*'))
    all_paths = [str(path) for path in all_paths]
    random.shuffle(all_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    img_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_paths]

    for path in all_paths:
        img = load_img(path)
        train_img.append(img)

    img_labels = np.asarray(img_labels)
    return train_img, img_labels

"""
    for classname in dataset.keys():
        print("Reading images for class {0}".format(classname))
        for img in dataset[classname]['training']:
            path = "../dataset-resized/" + classname + "/" + img
            # print("reading {0}".format(path))
            img = read_tensor_from_image_file(path, 224,224,0,255)
            train_img.append(img)
            train_label.append(label_id)

        for img in dataset[classname]['testing']:
            path = "../dataset-resized/" + classname + "/" + img
            # print("reading {0}".format(path))
            img = read_tensor_from_image_file(path, 224,224,0,255)
            test_img.append(img)
            test_label.append(label_id)

        for img in dataset[classname]['validation']:
            path = "../dataset-resized/" + classname + "/" + img
            # print("reading {0}".format(path))
            img = read_tensor_from_image_file(path, 224,224,0,255)
            val_img.append(img)
            val_label.append(label_id)

        label_id = label_id+1
    return (train_img, train_label, test_img, test_label, val_img, val_label)
"""

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      An OrderedDict containing an entry for each label subfolder, with images
      split into training, testing, and validation sets within each label.
      The order of items defines the class indices.
    """
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                                for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png']))
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def fire_module(input, fire_id, channel, s1, e1, e3,):
    """
    Basic module that makes up the SqueezeNet architecture. It has two layers.
     1. Squeeze layer (1x1 convolutions)
     2. Expand layer (1x1 and 3x3 convolutions)
    :param input: Tensorflow tensor
    :param fire_id: Variable scope name
    :param channel: Depth of the previous output
    :param s1: Number of filters for squeeze 1x1 layer
    :param e1: Number of filters for expand 1x1 layer
    :param e3: Number of filters for expand 3x3 layer
    :return: Tensorflow tensor
    """

    fire_weights = {'conv_s_1': tf.Variable(tf.truncated_normal([1, 1, channel, s1])),
                    'conv_e_1': tf.Variable(tf.truncated_normal([1, 1, s1, e1])),
                    'conv_e_3': tf.Variable(tf.truncated_normal([3, 3, s1, e3]))}

    fire_biases = {'conv_s_1': tf.Variable(tf.truncated_normal([s1])),
                   'conv_e_1': tf.Variable(tf.truncated_normal([e1])),
                   'conv_e_3': tf.Variable(tf.truncated_normal([e3]))}

    with tf.name_scope(fire_id):
        output = tf.nn.conv2d(input, fire_weights['conv_s_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_s_1')
        output = tf.nn.relu(tf.nn.bias_add(output, fire_biases['conv_s_1']))

        expand1 = tf.nn.conv2d(output, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1'])

        expand3 = tf.nn.conv2d(output, fire_weights['conv_e_3'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_3')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
        return tf.nn.relu(result)


def squeeze_net(input, classes):
    """
    SqueezeNet model written in tensorflow. It provides AlexNet level accuracy with 50x fewer parameters
    and smaller model size.
    :param input: Input tensor (4D)
    :param classes: number of classes for classification
    :return: Tensorflow tensor
    """

    weights = {'conv1': tf.Variable(tf.truncated_normal([7, 7, 3, 96])),
               'conv10': tf.Variable(tf.truncated_normal([1, 1, 512, classes]))}

    biases = {'conv1': tf.Variable(tf.truncated_normal([96])),
              'conv10': tf.Variable(tf.truncated_normal([classes]))}

    output = tf.nn.conv2d(input, weights['conv1'], strides=[1,2,2,1], padding='SAME', name='conv1')
    output = tf.nn.bias_add(output, biases['conv1'])

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')

    output = fire_module(output, s1=16, e1=64, e3=64, channel=96, fire_id='fire2')
    output = fire_module(output, s1=16, e1=64, e3=64, channel=128, fire_id='fire3')
    output = fire_module(output, s1=32, e1=128, e3=128, channel=128, fire_id='fire4')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')

    output = fire_module(output, s1=32, e1=128, e3=128, channel=256, fire_id='fire5')
    output = fire_module(output, s1=48, e1=192, e3=192, channel=256, fire_id='fire6')
    output = fire_module(output, s1=48, e1=192, e3=192, channel=384, fire_id='fire7')
    output = fire_module(output, s1=64, e1=256, e3=256, channel=384, fire_id='fire8')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')

    output = fire_module(output, s1=64, e1=256, e3=256, channel=512, fire_id='fire9')

    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout9')

    output = tf.nn.conv2d(output, weights['conv10'], strides=[1, 1, 1, 1], padding='SAME', name='conv10')
    output = tf.nn.bias_add(output, biases['conv10'])

    output = tf.nn.avg_pool(output, ksize=[1, 13, 13, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool10')

    return output

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

    return result

if __name__ == "__main__":
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=36906, stdoutToServer = True, stderrToServer = True)
    TESTING_PERCENTAGE = 10
    VALIDATION_PERCENTAGE = 10
    image_lists = create_image_lists("../dataset-resized", TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)

    train_img, train_label = loadDataset(image_lists)
    class_count = len(image_lists.keys())
    model = SqueezeNet()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_img[0], train_label[0], batch_size=1, epochs=5)

    #test_loss, test_acc = model.evaluate(image_lists['training'], test_labels)

    #print(model)
