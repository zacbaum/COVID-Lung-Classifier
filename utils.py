import keras
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


"""
CALLBACKS
"""

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
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
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
        plt.clf()
        pred = np.round(np.squeeze(self.model.predict(self.X_val)))
        true = self.Y_val
        cnf_mat = confusion_matrix(true, pred)
   
        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]
        print()
        print(cnf_mat)
        print()
