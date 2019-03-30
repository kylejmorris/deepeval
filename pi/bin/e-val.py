#!/usr/bin/env/python

import logging
import numpy as np
import os
import sys
import tensorflow as tf
from io import BytesIO
from os.path import expanduser
from picamera import PiCamera
from time import sleep, time
from VL53L0X import VL53L0X


LOG_FILE = os.environ["LOG_FILE"]
COOLDOWN = 3
SIZE = (224, 224)
DISTANCE_RANGE = range(100, 1000)
TRASH_LABEL = 5
INPUT_OP = "import/Placeholder"
OUTPUT_OP = "import/final_result"

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)-15s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger("e-val")

def camera():
    """Load the camera."""
    cam = PiCamera()
    cam.start_preview(alpha=128)
    LOGGER.info("Loaded camera")
    return cam


def model(path):
    """Load the model."""
    g = tf.Graph()
    gd = tf.GraphDef()
    with open(path, "rb") as f:
        gd.ParseFromString(f.read())
    with g.as_default():
        tf.import_graph_def(gd)
    return g, g.get_operation_by_name(INPUT_OP), g.get_operation_by_name(OUTPUT_OP)


def proximity():
    """Load the proximity sensor."""
    prox = VL53L0X()
    prox.start_ranging()
    LOGGER.info("Loaded proximity sensor")
    return prox


def ready():
    # TODO: Light an LED.
    LOGGER.info("Ready")


def wait(prox):
    """Wait for an object to come near."""
    while prox.get_distance() not in DISTANCE_RANGE:
        sleep(0.1)


def capture(sess, cam):
    """Capture an image."""
    buf = BytesIO()
    cam.capture(buf, format="jpeg", resize=SIZE)
    buf.seek(0)
    img = tf.io.decode_jpeg(buf.read(), channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, 0)
    img = tf.image.resize_bilinear(img, list(SIZE))
    # TODO: Do we need this?
    # img = tf.divide(tf.subtract(img, [input_mean]), [input_std])
    img = sess.run(img)
    LOGGER.info("Captured image")
    return img


def infer(sess, inp_op, out_op, img):
    """Run inference on an image."""
    pred = sess.run(out_op.outputs[0], {inp_op.outputs[0]: img})
    LOGGER.info("Results: " + str(pred))
    return pred


def interpret(out):
    """Interpret an inference result."""
    pred = out.argmax() != TRASH_LABEL
    LOGGER.info("Prediction: " + str(pred))
    return pred


def indicate(pred):
    """Indicate a prediction."""
    LOGGER.info(pred)  # TODO: Blink an LED.


if __name__ == "__main__":
    image_sess = tf.Session()
    cam = camera()
    prox = proximity()
    graph, inp_op, out_op = model(sys.arg[1] if len(sys.argv) > 1 else expanduser("~/model.pb"))
    infer_sess = tf.Session(graph=graph)
    # Warmup
    img = capture(image_sess, cam)
    infer(infer_sess, inp_op, out_op, img)
    ready()

    while True:
        wait(prox)
        LOGGER.info("Received trigger")
        start = time()
        img = capture(image_sess, cam)
        out = infer(infer_sess, inp_op, out_op, img)
        pred = interpret(out)
        indicate(pred)
        LOGGER.info("Time: " + str(time() - start))
        sleep(COOLDOWN)
