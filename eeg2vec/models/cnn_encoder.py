from torch import nn

class CNNEncoder(nn.Module):
    def __init__(self, conv_size, out_size, kernel_size_1, kernel_size_2):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=conv_size, kernel_size=kernel_size_1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_1),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=conv_size, out_channels=out_size, kernel_size=kernel_size_2),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return x
    
    
