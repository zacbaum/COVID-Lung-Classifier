import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical
import tempfile

from utils import PlotLosses, ConfusionMatrixPlotter

def add_l1l2_regularizer(model, l1=0.0, l2=0.0, reg_attributes=None):
    if not reg_attributes:
        reg_attributes = ['kernel_regularizer', 'bias_regularizer',
                          'beta_regularizer', 'gamma_regularizer']
    if isinstance(reg_attributes, str):
        reg_attributes = [reg_attributes]

    regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp.h5')
    model.save_weights(tmp_weights_path)

    # Reload the model
    model = model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)

    return model

batch_size = 256

input_shape = (int(720 / 8), int(1080 / 8), 1)

data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-cls"

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
)
train_flow = train_datagen.flow_from_directory(
    directory=os.path.join(data_folder, "train"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True,
)
test_datagen = ImageDataGenerator()
test_flow = test_datagen.flow_from_directory(
    directory=os.path.join(data_folder, "test"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
)

model = tf.keras.applications.VGG16(
    include_top=False, weights=None, input_shape=input_shape, classes=2
)

updated_model = Sequential()
for layer in model.layers:
    updated_model.add(layer)
updated_model.add(Flatten())
updated_model.add(Dense(512, activation='relu'))
updated_model.add(Dropout(0.85))
updated_model.add(Dense(512, activation='relu'))
updated_model.add(Dropout(0.85))
updated_model.add(Dense(1, activation='sigmoid'))
model = updated_model

#model = add_l1l2_regularizer(model, l2=0.00001, reg_attributes='kernel_regularizer')

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
    ],
)

train_steps = train_flow.n // train_flow.batch_size
valid_steps = test_flow.n // test_flow.batch_size

plot_losses = PlotLosses()

cm_flow = test_datagen.flow_from_directory(
    directory=os.path.join(data_folder, "test"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=1,
    class_mode="binary",
    shuffle=False,
)
x, y = zip(*(cm_flow[i] for i in range(0, len(cm_flow), 2)))
x_val, y_val = np.vstack(x), np.vstack(to_categorical(y))[:,1]
plot_cm = ConfusionMatrixPlotter(x_val, y_val, normalize=True)

history = model.fit_generator(
    train_flow,
    train_steps,
    epochs=100,
    validation_data=test_flow,
    validation_steps=valid_steps,
    callbacks=[
        plot_losses,
        plot_cm,
    ],
    verbose=2,
    max_queue_size=250,
    workers=32,
)
