import torch
import matplotlib.pyplot as plt
import numpy as np

import os
import zipfile

from pathlib import Path

import requests

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
  
  return ACC



def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

