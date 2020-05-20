import cv2
import numpy as np
from threading import Event, Thread
import time
import tensorflow.keras
from PIL import Image, ImageOps
import json
from tensorflow.keras.models import model_from_json


p = 0
prediction = ['1 Finger', '2 Fingers',
              '3 Fingers',  '4 Fingers', '5 Fingers']

# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

# parameters
bgCapture = 0
bgSubThreshold = 50
learningRate = 0
blurValue = 41
threshold = 60
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
imgCount = 0

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model

json_file = open('Hand_gesture/Saved_models/final1', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('Hand_gesture/final_model1.h5')


class RepeatedTimer:

    def __init__(self, interval):
        self.interval = interval
        self.start = time.time()
        self.event = Event()
        self.thread = Thread(target=self._target)
        self.thread.start()

    def _target(self):
        while not self.event.wait(self._time):
            print("Current prediction:-" + str(p))

    @property
    def _time(self):
        return self.interval - ((time.time() - self.start) % self.interval)

    def stop(self):
        self.event.set()
        self.thread.join()


# start timer
timer = RepeatedTimer(2)


def pred():
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('img0.png')
    image = image.convert('RGB')

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Load the image into the array
    data[0] = image_array

    # run the inference
    prediction = loaded_model.predict(data)
    return np.argmax(prediction)


def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    #fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('original', frame)

    if bgCapture == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]): frame.shape[1]]  # clip the ROI

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(
            blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(f'img0.png', thresh)
        p = pred()
        """ print('predecting: - ', p) """
        cv2.putText(thresh, prediction[p], (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('threshold', thresh)

    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(
            0, bgSubThreshold, detectShadows=False)
        bgCapture = 1

timer.stop()
