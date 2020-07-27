from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
import os
import tensorflow as tf

from utils import PlotLosses, ConfusionMatrixPlotter

batch_size = 256

input_shape = (int(1080 / 4), int(720 / 4), 1)

data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-uk"

test_datagen = ImageDataGenerator(zoom_range=[0.75, 0.75],)
test_flow = test_datagen.flow_from_directory(
    directory=os.path.join(data_folder, "test"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=1,
    class_mode="binary",
    shuffle=False,
)

x, y = zip(*(test_flow[i] for i in range(0, len(test_flow), 1)))
x_val, y_val = np.vstack(x), np.vstack(to_categorical(y))[:, 1]

print("ASSESS QUALITY...\n")

model = load_model("quality-cls-100to1.h5", compile=False)
preds = model.predict(x_val, batch_size=256, max_queue_size=250, workers=32,)
preds = np.round(np.squeeze(preds))

x_good = [x_val[i] for i, x in enumerate(preds) if x == 1]
x_poor = [x_val[i] for i, x in enumerate(preds) if x == 0]

for i, im in enumerate(x_good):
    cv2.imwrite(
        os.path.join("./result-100to1-keep/" + str(i) + ".png"),
        cv2.cvtColor(im, cv2.COLOR_GRAY2RGB),
    )

for i, im in enumerate(x_poor):
    cv2.imwrite(
        os.path.join("./result-100to1-remove/" + str(i) + ".png"),
        cv2.cvtColor(im, cv2.COLOR_GRAY2RGB),
    )

print("\nPREDICT (+) vs (-) ON GOOD IMAGES")

x_good = np.array(x_good)

for fold in ["fold1", "fold2", "fold3", "fold4", "fold5"]:

    print("\n===================================================")
    print("                       " + fold + "                       ")
    print("===================================================")

    model = load_model(fold + ".h5", compile=False)

    preds = model.predict(x_good, batch_size=256, max_queue_size=250, workers=32,)

    preds = np.round(np.squeeze(preds))
    print(sum(preds) / len(preds))
