import os
import torch
from torchmetrics import Accuracy
import utils
from load_dataset import CustomImageDataset
from nn_model import NeuralNetwork
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

# Load label dictionary from config file
list_label = utils.label_dict_from_config_file("generate_data/hand_gesture.yaml")
print(f"Label dictionary: {list_label}")

# Get label names from the dictionary (values are the gesture labels)
label_names = list(list_label.values())

# Load test dataset
DATA_FOLDER_PATH = "./generate_data/gesture_data/"
test_dataset = CustomImageDataset(os.path.join(DATA_FOLDER_PATH, "landmark_test.csv"))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)

# Load model
best_model_path = "./models/models_22-06 19:12_NeuralNetwork_best"
network = NeuralNetwork().to(device)  # Send the model to the device (GPU or CPU)
network.load_state_dict(torch.load(best_model_path, map_location=device))
network.eval()  # Set model to evaluation mode

# Initialize accuracy metric
acc_test = Accuracy(num_classes=len(list_label), task='multiclass').to(device)

# Lists for confusion matrix
all_preds = []
all_labels = []

# Test loop for evaluation
for i, (test_input, test_label) in enumerate(test_loader):
    test_input, test_label = test_input.to(device), test_label.to(device)
    with torch.no_grad():  # Disable gradient calculation during inference
        logits = network(test_input)
        preds = torch.argmax(logits, dim=1)  # Get predictions (index of max logit)

    acc_test.update(preds, test_label)  # Update accuracy metric

    # Append predictions and true labels for confusion matrix
    all_preds.extend(preds.cpu().numpy())  # Move data to CPU for further processing
    all_labels.extend(test_label.cpu().numpy())

# Print accuracy
print(network.__class__.__name__)
print(f"Accuracy of model: {acc_test.compute().item():.4f}")
print("========================================================================")

# Confusion matrix
os.makedirs("./confusion_matrix", exist_ok=True)  # Ensure the output directory exists
cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
save_path = "./confusion_matrix/confusion_matrix.png"
plt.savefig(save_path)  # Save the confusion matrix plot
print(f"Confusion matrix saved to {save_path}")

# Classification report
report = classification_report(all_labels, all_preds, target_names=label_names)  # Generate the classification report

# Save the classification report to a text file
os.makedirs("./report", exist_ok=True)  # Ensure the report directory exists
report_path = "./report/classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to {report_path}")
