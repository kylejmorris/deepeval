#!/usr/bin/env/python

import numpy as np
import os
import sys
import tensorflow as tf
from io import BytesIO
from picamera import PiCamera
from PIL import Image
from time import sleep
from VL53L0X import VL53L0X


COOLDOWN = 5
IMAGE_BUFFER = BytesIO()
SIZE = (224, 224)
DISTANCE_RANGE = range(100, 1000)
TRASH_LABEL = 5


def camera():
    """Load the camera."""
    cam = PiCamera()
    cam.start_preview(alpha=128)
    print("Loaded camera")
    return cam


def model(path):
    """Load the model."""
    gd = tf.GraphDef()
    with tf.gfile.FastGFile(path, "rb") as f:
        gd.ParseFromString(f.read())
    return tf.Session(), tf.import_graph_def(gd, return_elements=["final_output"])[0]


def proximity():
    """Load the proximity sensor."""
    prox = VL53L0X()
    prox.start_ranging()
    print("Loaded proximity sensor")
    return prox


def ready():
    # TODO: Light an LED.
    print("Ready")


def wait(prox):
    """Wait for an object to come near."""
    while prox.get_distance() not in DISTANCE_RANGE:
        sleep(0.1)


def capture(cam, buf):
    """Capture an image."""
    cam.capture(buf, format="png", resize=SIZE)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"), dtype=float)
    print("Captured image")
    return img


def infer(sess, op, img):
    """Run inference on an image."""
    return sess.run(op, img)


def interpret(out):
    """Interpret an inference result."""
    pred = out != TRASH_LABEL
    print("Prediction: " + str(pred))
    return pred


def indicate(pred):
    """Indicate a prediction."""
    print(pred)  # TODO: Blink an LED.


if __name__ == "__main__":
    cam = camera()
    prox = proximity()
    sess, op = model(sys.arg[1] if len(sys.argv) > 1 else expanduser("~/model.pb"))
    ready()
    while True:
        wait(prox)
        img = capture(cam, IMAGE_BUFFER)
        out = infer(sess, op, img)
        pred = interpret(out)
        indicate(pred)
        sleep(COOLDOWN)
