import os
import torch
from torchmetrics import Accuracy
import utils
from load_dataset import CustomImageDataset
from nn_model import NeuralNetwork
from datetime import datetime
from torch import optim
import logging
from  torch import nn

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Directory paths and model settings
DATA_DIR_PATH = "./generate_data/gesture_data/"
LIST_LABEL = utils.label_dict_from_config_file("generate_data/hand_gesture.yaml")
train_path = os.path.join(DATA_DIR_PATH, "landmark_train.csv")
val_path = os.path.join(DATA_DIR_PATH, "landmark_val.csv")
save_path = "models"
os.makedirs(save_path, exist_ok=True)

# Load datasets and dataloaders
train_dataset = CustomImageDataset(train_path)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)

val_dataset = CustomImageDataset(val_path)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)

# Initialize model, loss function, optimizer, and early stopper
model = NeuralNetwork().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
early_stopper = utils.EarlyStopper(patience=30, min_delta=0.01)

def train(train_data_loader, val_data_loader, model, loss_function, early_stopper, optimizer):
    best_vloss = float('inf')
    timestamp = datetime.now().strftime('%d-%m %H:%M')

    for epoch in range(300):
        # Training step
        model.train()
        running_loss = 0.0
        acc_train = Accuracy(num_classes=len(LIST_LABEL), task='multiclass').to(device)

        for batch_number, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(inputs)  # forward pass
            loss = loss_function(preds, labels)
            loss.backward()
            optimizer.step()

            acc_train.update(model.predict_with_known_class(inputs), labels)
            running_loss += loss.item()

        avg_loss = running_loss / len(train_data_loader)

        # Validation step
        model.eval()
        running_vloss = 0.0
        acc_val = Accuracy(num_classes=len(LIST_LABEL), task='multiclass').to(device)

        with torch.no_grad():
            for vdata in val_data_loader:
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                preds = model(vinputs)
                vloss = loss_function(preds, vlabels)
                running_vloss += vloss.item()
                acc_val.update(model.predict_with_known_class(vinputs), vlabels)

        avg_vloss = running_vloss / len(val_data_loader)
        logging.info(f"Epoch {epoch}: ")
        logging.info(f"Training accuracy: {acc_train.compute().item():.4f}, Validation accuracy: {acc_val.compute().item():.4f}")
        logging.info(f"Training loss: {avg_loss:.4f}, Validation loss: {avg_vloss:.4f}")

        # Track best model and save the best model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_path = f'./{save_path}/models_{timestamp}_{model.__class__.__name__}_best.pth'
            torch.save(model.state_dict(), best_model_path)

        if early_stopper.early_stop(avg_vloss):
            logging.info(f"Early stopping at epoch {epoch}, minimum loss: {early_stopper.watched_metrics}")
            break

    model_path = f'./{save_path}/models_{timestamp}_{model.__class__.__name__}_last.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f"Final validation accuracy: {acc_val.compute().item():.4f}")

    return model, best_model_path

# Start training
model, best_model_path = train(train_data_loader, val_data_loader, model, loss_function, early_stopper, optimizer)
