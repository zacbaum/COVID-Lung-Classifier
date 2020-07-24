from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.cm as cm
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import activations
from vis.visualization import visualize_saliency, overlay, visualize_cam
from vis.utils import utils
import tempfile

output_folder_path = os.path.join(os.getcwd(), "result_sm-cam")
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

input_shape = (int(1080 / 4), int(720 / 4), 1)

data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-cv/"

fold = "fold5"

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
x, y = zip(*(attention_flow[i] for i in range(0, len(attention_flow), 50)))
x_val, y_val = np.vstack(x), np.vstack(to_categorical(y))[:, 1]

model = load_model(fold + ".h5")
if fold == "fold1": layer_idx = utils.find_layer_idx(model, "dense_1")
if fold == "fold2": layer_idx = utils.find_layer_idx(model, "dense_3")
if fold == "fold3": layer_idx = utils.find_layer_idx(model, "dense_5")
if fold == "fold4": layer_idx = utils.find_layer_idx(model, "dense_7")
if fold == "fold5": layer_idx = utils.find_layer_idx(model, "dense_9")
model.layers[layer_idx].activation = activations.linear
model_path = os.path.join(
    tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".h5"
)
try:
    model.save(model_path)
    model = load_model(model_path)
finally:
    os.remove(model_path)

for i in range(len(x_val)):
    print(i, len(x_val))
    im = x_val[i]
    label = int(y_val[i])

    grads = visualize_saliency(
        model,
        layer_idx,
        filter_indices=None,
        seed_input=im,
        backprop_modifier="guided",
    )
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)            
    cv2.imwrite(
        os.path.join(output_folder_path, fold + "-sm-" + str(i) + "-class-" + str(label) + ".png"), 
        jet_heatmap
    )
    '''
    grads = visualize_cam(
        model,
        layer_idx,
        filter_indices=None,
        seed_input=im,
        backprop_modifier="guided",
        penultimate_layer_idx=utils.find_layer_idx(model, "block5_pool"),
    )
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        os.path.join(output_folder_path, fold + "-cam-" + str(i) + "-class-" + str(label) + ".png"),
        overlay(jet_heatmap, cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
    )
    '''