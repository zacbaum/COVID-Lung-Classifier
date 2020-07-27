from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from utils import plotAttention
from vis.utils import utils
import numpy as np
import os
import random
import tempfile
import tensorflow as tf

output_folder_path = os.path.join(os.getcwd(), "result_sm-cam")
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

input_shape = (int(1080 / 4), int(720 / 4), 1)

data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-cv/"

fold = "fold1"

fold_folder = os.path.join(data_folder, fold)

attention_datagen = ImageDataGenerator(zoom_range=[0.75, 0.75],)
attention_flow = attention_datagen.flow_from_directory(
    directory=os.path.join(fold_folder, "test"),
    target_size=(input_shape[0], input_shape[1]),
    color_mode="grayscale",
    batch_size=1,
    class_mode="binary",
    shuffle=False,
)
x, y = zip(*(attention_flow[i] for i in range(0, len(attention_flow), 1)))
x_val, y_val = np.vstack(x), np.vstack(to_categorical(y))[:, 1]

model = load_model(fold + ".h5")

preds = model.predict(x_val, batch_size=256, max_queue_size=250, workers=32, verbose=1,)
preds = np.round(np.squeeze(preds))

if fold == "fold1":
    layer_idx = utils.find_layer_idx(model, "dense_1")
if fold == "fold2":
    layer_idx = utils.find_layer_idx(model, "dense_3")
if fold == "fold3":
    layer_idx = utils.find_layer_idx(model, "dense_5")
if fold == "fold4":
    layer_idx = utils.find_layer_idx(model, "dense_7")
if fold == "fold5":
    layer_idx = utils.find_layer_idx(model, "dense_9")
model.layers[layer_idx].activation = activations.linear
model_path = os.path.join(
    tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".h5"
)
try:
    model.save(model_path)
    model = load_model(model_path)
finally:
    os.remove(model_path)

TP = random.sample([i for i, x in enumerate(y_val) if (x == 1 and preds[i] == 1)], 5)
TN = random.sample([i for i, x in enumerate(y_val) if (x == 0 and preds[i] == 0)], 5)
FN = random.sample([i for i, x in enumerate(y_val) if (x == 1 and preds[i] == 0)], 5)
FP = random.sample([i for i, x in enumerate(y_val) if (x == 0 and preds[i] == 1)], 5)

for i in TP:
    im = x_val[i]
    plotAttention(model, layer_idx, im, i, "TP", fold, output_folder_path)
print("Created CAMs and Saliency Maps for TP images: " + str(TP))

for i in TN:
    im = x_val[i]
    plotAttention(model, layer_idx, im, i, "TN", fold, output_folder_path)
print("Created CAMs and Saliency Maps for TN images: " + str(TN))

for i in FN:
    im = x_val[i]
    plotAttention(model, layer_idx, im, i, "FN", fold, output_folder_path)
print("Created CAMs and Saliency Maps for FN images: " + str(FN))

for i in FP:
    im = x_val[i]
    plotAttention(model, layer_idx, im, i, "FP", fold, output_folder_path)
print("Created CAMs and Saliency Maps for FP images: " + str(FP))
