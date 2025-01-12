import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEmbeddingModel(nn.Module):
    def __init__(
        self,
        input_channels: int,            # Number of EEG input channels
        input_length: int,              # Number of time steps per channel
        conv_filters: list,             # List of filter sizes for Conv1d layers
        kernel_sizes: list,             # List of kernel sizes for Conv1d layers
        pooling_factor: int,            # Downsampling factor for MaxPool1d
        embedding_dim: int,             # Size of the final embedding
    ):
        super(EEGEmbeddingModel, self).__init__()
        
        assert len(conv_filters) == len(kernel_sizes), "conv_filters and kernel_sizes must have the same length."
        
        # Convolutional Layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            in_channels = out_channels  # Update for the next layer
        
        self.pooling = nn.MaxPool1d(kernel_size=pooling_factor)

        output_size = input_length // pooling_factor
        # Fully Connected Layers for Final Embedding
        self.fc1 = nn.Linear(output_size, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
    
    def forward(self, x):
        # x shape: (batch_size, input_channels, input_length)
        for conv in self.conv_layers:
            x = F.relu(conv(x))  # Apply each convolution + ReLU
        
        x = self.pooling(x)  # Apply max pooling
        
        # Pooling over the sequence length
        x = x.mean(dim=1)  # Global mean pooling
        x = F.relu(self.fc1(x))  # Fully connected layer + ReLU
        x = self.fc2(x)  # Final embedding
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, z_i, z_j, label):
        # Compute L2 distance between embeddings
        distance = F.pairwise_distance(z_i, z_j)
        
        # Contrastive loss formula
        loss = label * (distance ** 2) + (1 - label) * F.relu(self.margin - distance) ** 2
        return loss.mean()

# Example pair sampling
def create_pairs(data, labels):
    pairs = []
    pair_labels = []
    num_samples = len(data)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            pairs.append((data[i], data[j]))
            pair_labels.append(1 if labels[i] == labels[j] else 0)
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs]), torch.tensor(pair_labels)


model = EEGEmbeddingModel(1,500,[16,32,16],[3,5,3], 1,32)

# Example usage

# Generate random data
data = torch.randn(10, 1, 500)
labels = torch.randint(0, 2, (10,))

# Create pairs
x_i, x_j, pair_labels = create_pairs(data, labels)
print(x_i.shape, x_j.shape, pair_labels.shape)

# Forward pass
z_i = model(x_i)
z_j = model(x_j)

# Compute loss
criterion = ContrastiveLoss()
loss = criterion(z_i, z_j, pair_labels)
print(loss)