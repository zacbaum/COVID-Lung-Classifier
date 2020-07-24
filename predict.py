import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from utils import PlotLosses, ConfusionMatrixPlotter

batch_size = 256

input_shape = (int(1080 / 4), int(720 / 4), 1)

data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-uk"

test_datagen = ImageDataGenerator(
    zoom_range=[0.75, 0.75],
)
test_flow = test_datagen.flow_from_directory(
    directory=os.path.join(data_folder, "test"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
)

print("ASSESS QUALITY")

for qual in ["quality-cls-50to1", "quality-cls-100to1"]:

    model = load_model(qual + ".h5", compile=False)

    preds = model.predict(
        test_flow, 
        max_queue_size=250, 
        workers=32,
        verbose=1,
    )

    preds = np.round(preds)
    print(sum(preds) / len(preds))

print("PREDICT (+) vs (-)")

for fold in ["fold1", "fold2", "fold3", "fold4", "fold5"]:

    model = load_model(fold + ".h5", compile=False)

    preds = model.predict(
        test_flow, 
        max_queue_size=250, 
        workers=32,
        verbose=1,
    )

    preds = np.round(preds)
    print(sum(preds) / len(preds))