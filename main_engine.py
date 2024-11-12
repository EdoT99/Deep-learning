import torch



def train_loop(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               epochs: int,
               accuracy_fn,
               device: torch.device = device
               ):
  '''
  Trains a PyTorch MOdel on train data
  '''
  train_acc, loss_train = 0, 0
  model.to(device)
  model.train()

  for batch, (x, y) in enumerate(train_dataloader):
    
    X,Y = x.to(device), y.to(device)

    y_logits = model(x)
    loss = loss_fn(y_logits, y)
    loss_train += loss
    train_acc += accuracy_fn(y_true=y,y_pred=y_logits.argmax(dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

  loss_train /= len(train_dataloader)
  train_acc /= len(train_dataloader)
  print(f"Train loss: {loss} | train accuracy: {train_acc}")
  return train_acc,loss



def test_loop(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
  '''
  Tests and prints a model's loss and accuracy on a testing dataset
  '''
  model.to(device)
  model.eval()

  with torch.inference_mode():
    test_loss, test_acc = 0,0
    for batch, (x, y) in enumerate(test_dataloader):
      X,Y = x.to(device), y.to(device)
      test_pred = model(x)
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y_true=y,y_pred=test_pred.argmax(dim=1))

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print(f'Test loss: {test_loss} | Test accuracy: {test_acc}')
  return {'Model': model.__class__.__name__, 'Acc': test_acc ,'Test_loss': test_loss}







def training_season(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) :      #looping through training and testing data for n epochs, powered by duaLIpa
  
  for epoch in epochs:
    train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
    test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
  results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
  
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
  
    # Return the filled results at the end of the epochs
    return results
