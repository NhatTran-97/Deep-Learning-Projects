
import yaml
import torch

from  torch import nn
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        """ This class aims to build a Neural Network Classification. This model contains 4 hidden layer
        *  Input: 63 Features
        *  Hidden Layer 1: (63, 128), (Relu, BatchNorm1d) 
        *  Hidden Layer 1: (128, 128) (Relu, Dropout:0.4) 
        *  Hidden Layer 2: (128, 128) (Relu, Dropout:0.4)  
        *  Hidden Layer 3: (128, 128) (Relu, Dropout:0.6)  
        *  Output: 128: 128 of Number of Labels 
        """
        self.flatten = nn.Flatten()
        list_label = utils.label_dict_from_config_file('generate_data/hand_gesture.yaml')
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            nn.Linear(128, len(list_label))
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) # Output of last layer
        return logits
    def predict(self, x, threshold=0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob, dim=1)
        return torch.where(softmax_prob[0, chosen_ind] > threshold, chosen_ind, -1)
    
    def predict_with_known_class(self, x):
        
        x = x.to(next(self.parameters()).device)  
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits) 
        return torch.argmax(softmax_prob, dim=1)
    
    def score(self, logits):
        return -torch.amax(logits, dim=1)

