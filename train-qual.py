import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

from utils import PlotLosses, ConfusionMatrixPlotter

batch_size = 256

input_shape = (int(1080 / 4), int(720 / 4), 1)

data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-qual"

train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=[0.60, 0.75],
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode="constant",
    cval=0,
)
train_flow = train_datagen.flow_from_directory(
    directory=os.path.join(data_folder, "train"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True,
)

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

model = tf.keras.applications.VGG16(
    include_top=False, 
    weights=None, 
    input_shape=input_shape, 
    classes=2
)
updated_model = Sequential()
for layer in model.layers:
    updated_model.add(layer)
updated_model.add(Flatten())
updated_model.add(Dense(512, activation="relu"))
updated_model.add(Dropout(0.5))
updated_model.add(Dense(1, activation="sigmoid"))
model = updated_model
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy",],
)

cm_flow = test_datagen.flow_from_directory(
    directory=os.path.join(data_folder, "test"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=1,
    class_mode="binary",
    shuffle=False,
)
x, y = zip(*(cm_flow[i] for i in range(0, len(cm_flow), 1)))
x_val, y_val = np.vstack(x), np.vstack(to_categorical(y))[:, 1]
plot_cm = ConfusionMatrixPlotter(x_val, y_val, normalize=False)

history = model.fit_generator(
    train_flow,
    train_flow.n // train_flow.batch_size,
    epochs=50,
    validation_data=test_flow,
    validation_steps=test_flow.n // test_flow.batch_size,
    class_weight={0: 100., 1: 1.},
    callbacks=[plot_cm,],
    verbose=2,
    max_queue_size=250,
    workers=32,
)

model.save("quality-cls-100to1.h5")
