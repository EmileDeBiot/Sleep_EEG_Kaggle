from torch import nn

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=2, stride=1),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return x
    
    
