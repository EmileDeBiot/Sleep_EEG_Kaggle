from torch import nn

class CNNEncoder(nn.Module):
    def __init__(self, conv_sizes, kernel_sizes):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential()
        for i, (conv_size, kernel_size) in enumerate(zip(conv_sizes, kernel_sizes)):
            if i == 0:
                self.conv_layers.add_module(f'conv_{i}', nn.Conv1d(in_channels=5, out_channels=conv_size, kernel_size=kernel_size))
            else:
                self.conv_layers.add_module(f'conv_{i}', nn.Conv1d(in_channels=conv_sizes[i-1], out_channels=conv_size, kernel_size=kernel_size))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
            self.conv_layers.add_module(f'pool_{i}', nn.MaxPool1d(kernel_size=2))
        self.conv_layers.add_module('flatten', nn.Flatten())

    def forward(self, x):
        x = self.conv_layers(x)
        return x
    
    
