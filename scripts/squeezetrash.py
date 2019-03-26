import tensorflow as tf
from tensorflow import keras
import squeezenet

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

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
    img = read_tensor_from_image_file(
        "/deepeval/testing_images/glass/0.jpeg",
        input_height=224,
        input_width=224,
        input_mean=0,
        input_std=255)

    model = squeezenet.squeeze_net(img, classes=6)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(model)

