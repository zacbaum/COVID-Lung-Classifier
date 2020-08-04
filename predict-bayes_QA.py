from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

batch_size = 256

input_shape = (int(1080 / 4), int(720 / 4), 1)

pr_d_of_g = 14067 / (14305 + 14067)
pr_cla = 14305 / (14305 + 14067)

print("\nPREDICT GOOD vs BAD IMAGES")

d_of_g = np.load("/home/zbaum/Baum/ALOCC-CVPR2018/samples/check_QA/50_0.0/d_of_g_values.npy")
thresh = -16.8
max_val = max(d_of_g)
min_val = min(d_of_g)
gap = max(abs(max_val - thresh), abs(thresh - min_val))
max_val = thresh + gap
min_val = thresh - gap
p_d_of_g = np.squeeze(np.array([(x - min_val)/(max_val - min_val) for x in d_of_g]))

data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-alocc/"
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
x_val = x_val[:d_of_g.shape[0]]
y_val = y_val[:d_of_g.shape[0]]

model = load_model("quality-cls-50to1.h5", compile=False)
preds = model.predict(x_val, batch_size=256, max_queue_size=250, workers=32,)
p_cla = np.squeeze(preds)

p_final = np.round(p_cla * pr_cla + p_d_of_g * pr_d_of_g)

cnf_mat = confusion_matrix(y_val, p_final)
cnf_mat = cnf_mat.astype("float") / cnf_mat.sum(axis=1)[:, np.newaxis]
print()
print(cnf_mat)

acc = accuracy_score(y_val, p_final)
print("\nCORRECTION RATE: {:4f}".format(acc))

print("===================================================")

print("\nPREDICT CROSS VALIDATION")

for fold in ["fold1", "fold2", "fold3", "fold4", "fold5"]:

    print("\n===================================================")
    print("                       " + fold + "                       ")
    print("===================================================")

    print("ASSESS QUALITY w/ ALOCC...")

    d_of_g = np.load("/home/zbaum/Baum/ALOCC-CVPR2018/samples/" + fold + "/50_0.0/d_of_g_values.npy")
    thresh = -16.8
    max_val = max(d_of_g)
    min_val = min(d_of_g)
    gap = max(abs(max_val - thresh), abs(thresh - min_val))
    max_val = thresh + gap
    min_val = thresh - gap
    p_d_of_g = np.squeeze(np.array([(x - min_val)/(max_val - min_val) for x in d_of_g]))

    print("ASSESS QUALITY w/ VGG...\n")

    data_folder = "/home/zbaum/Baum/COVID-Lung-Classifier/data-cv/" + fold
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
    x_val = x_val[:d_of_g.shape[0]]

    model = load_model("quality-cls-50to1.h5", compile=False)
    preds = model.predict(x_val, batch_size=256, max_queue_size=250, workers=32,)
    p_cla = np.squeeze(preds)

    p_final = np.round(p_cla * pr_cla + p_d_of_g * pr_d_of_g)

    x_good = [x_val[i] for i, x in enumerate(p_final) if x == 1]
    x_poor = [x_val[i] for i, x in enumerate(p_final) if x == 0]

    y_good = [y_val[i] for i, x in enumerate(p_final) if x == 1]
    y_poor = [y_val[i] for i, x in enumerate(p_final) if x == 0]

    print("Removed {} 'poor' quality images...".format(len(y_poor)))

    print("\nPREDICT (+) vs (-) ON GOOD IMAGES")
    x_good = np.array(x_good)

    model = load_model(fold + ".h5", compile=False)

    preds = model.predict(x_good, batch_size=256, max_queue_size=250, workers=32,)

    fpr, tpr, thresholds = roc_curve(y_good, preds)
    sensitivity = tpr
    specificity = 1 - fpr

    print("SENSITIVITY: 0.8")
    nearest_idx = np.abs(sensitivity - 0.8).argmin()
    print("SPECIFICITY IS {:4f} AT SENSITIVITY {:4f}; THRESH: {:4f}".format(specificity[nearest_idx], sensitivity[nearest_idx], thresholds[nearest_idx]))

    print("\nSENSITIVITY: 0.9")
    nearest_idx = np.abs(sensitivity - 0.9).argmin()
    print("SPECIFICITY IS {:4f} AT SENSITIVITY {:4f}; THRESH: {:4f}".format(specificity[nearest_idx], sensitivity[nearest_idx], thresholds[nearest_idx]))
    
    print("\nSENSITIVITY: 0.95")
    nearest_idx = np.abs(sensitivity - 0.95).argmin()
    print("SPECIFICITY IS {:4f} AT SENSITIVITY {:4f}; THRESH: {:4f}".format(specificity[nearest_idx], sensitivity[nearest_idx], thresholds[nearest_idx]))
    
    print("\nSPECIFICITY: 0.8")
    nearest_idx = np.abs(specificity - 0.8).argmin()
    print("SENSITIVITY IS {:4f} AT SPECIFICITY {:4f}; THRESH: {:4f}".format(sensitivity[nearest_idx], specificity[nearest_idx], thresholds[nearest_idx]))
    
    print("\nSPECIFICITY: 0.9")
    nearest_idx = np.abs(specificity - 0.9).argmin()
    print("SENSITIVITY IS {:4f} AT SPECIFICITY {:4f}; THRESH: {:4f}".format(sensitivity[nearest_idx], specificity[nearest_idx], thresholds[nearest_idx]))
    
    print("\nSPECIFICITY: 0.95")
    nearest_idx = np.abs(specificity - 0.95).argmin()
    print("SENSITIVITY IS {:4f} AT SPECIFICITY {:4f}; THRESH: {:4f}".format(sensitivity[nearest_idx], specificity[nearest_idx], thresholds[nearest_idx]))
    
    preds = np.round(np.squeeze(preds))
    cnf_mat = confusion_matrix(y_good, preds)
    cnf_mat = cnf_mat.astype("float") / cnf_mat.sum(axis=1)[:, np.newaxis]
    print()
    print(cnf_mat)

    acc = accuracy_score(y_good, preds)
    print("\nCORRECTION RATE: {:4f}".format(acc))