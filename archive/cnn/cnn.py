from torch import nn
import torch.nn.functional as F
import torch

class CNNclassifier(nn.Module):
    def __init__(self, num_channels, window_size, layer_sizes: list, kernel_sizes: list, num_classes: int):
        super(CNNclassifier, self).__init__()
        self.convs = nn.ModuleList()  # Store convolutional layers
        
        # Add convolutional layers
        for i in range(len(layer_sizes)):
            self.convs.append(nn.Conv1d(num_channels, layer_sizes[i], kernel_sizes[i]))
            num_channels = layer_sizes[i]  # Update number of channels for next layer

        # Fully connected layer
        self.fc = nn.Linear(layer_sizes[-1] * (window_size - sum(kernel_sizes) + len(kernel_sizes)), num_classes)

    def forward(self, x):
        # Pass input through convolutional layers
        for conv in self.convs:
            x = F.relu(conv(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, flattened_size)

        # Fully connected layer
        x = self.fc(x)  # Output: (batch_size, 5)
        return torch.sigmoid(x)  # Sigmoid for binary multi-label classification