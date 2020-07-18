import keras
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []

        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="v loss")
        ax1.legend()
        ax2.plot(self.x, self.acc, label="acc")
        ax2.plot(self.x, self.val_acc, label="v acc")
        ax2.legend()

        plt.show()
        plt.savefig("live_metrics.png", dpi=250)
        plt.close()


class ConfusionMatrixPlotter(keras.callbacks.Callback):
    """Plot the confusion matrix on a graph and update after each epoch
    # Arguments
        X_val: The input values 
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title
    """

    def __init__(self, X_val, Y_val, normalize=False):
        self.X_val = X_val
        self.Y_val = Y_val
        self.normalize = normalize

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pred = np.round(np.squeeze(self.model.predict(self.X_val)))
        true = self.Y_val
        cnf_mat = confusion_matrix(true, pred)

        if self.normalize:
            cnf_mat = cnf_mat.astype("float") / cnf_mat.sum(axis=1)[:, np.newaxis]
        print()
        print(cnf_mat)
        print()


def add_l1l2_regularizer(model, l1=0.0, l2=0.0, reg_attributes=None):
    if not reg_attributes:
        reg_attributes = [
            "kernel_regularizer",
            "bias_regularizer",
            "beta_regularizer",
            "gamma_regularizer",
        ]
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
    tmp_weights_path = os.path.join(tempfile.gettempdir(), "tmp.h5")
    model.save_weights(tmp_weights_path)

    # Reload the model
    model = model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)

    return model


def plotImages(images_arr, labels, name="train_sample"):
    fig, axes = plt.subplots(10, 10, figsize=(108, 72))
    axes = axes.flatten()
    for img, lab, ax in zip(images_arr, labels, axes):
        ax.imshow(img, cmap="gray")
        ax.set_title(lab)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(name + ".png")