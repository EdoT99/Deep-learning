import torch
import matplotlib.pyplot as plt
import numpy as np

#----plotting loss-functions------

def plot_losses(loss_values_train,loss_values_test,epochs_count):

  plt.figure(figsize=(16, 8))
  # Plot training and test loss curves
  plt.plot(epochs_count, loss_values_train, label="train loss")
  plt.plot(epochs_count, loss_values_test, label="test loss")

  plt.title('Training and Test Loss Curves')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()

#-----compute accuracy--------

def accuracy_fn(y_true,y_pred):
  correct = torch.eq(y_true,y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc
