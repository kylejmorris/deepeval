#!/usr/bin/env/python

import tensorflow as tf
from io import BytesIO
from picamera import PiCamera
from PIL import Image
from time import sleep
from VL53L0X import VL53L0X


COOLDOWN = 5000
IMAGE_BUFFER = BytesIO()
DISTANCE_THRESHOLD = 1000
TRASH_LABEL = 5


def camera():
    """Load the camera."""
    cam = PiCamera()
    cam.start_preview(alpha=128)
    print("Loaded camera")
    return cam


def model():
    """Load the model."""
    pass  # TODO: Actually load the model.


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
    while prox.get_distance() > DISTANCE_THRESHOLD:
        sleep(1)


def capture(cam, buf):
    """Capture an image."""
    cam.capture(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    print("Captured image")
    return img


def infer(mod, img):
    """Run inference on an image."""
    pass  # TODO: Run the model on the image.


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
    mod = model()
    ready()
    while True:
        wait(prox)
        img = capture(cam, IMAGE_BUFFER)
        out = infer(mod, img)
        pred = interpret(out)
        indicate(pred)
        sleep(COOLDOWN)
