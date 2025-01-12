import torch.nn as nn
from eeg2vec.models.cnn_encoder import CNNEncoder
class CNNTransformer(nn.Module):
    def __init__(self, transformer_d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, conv_sizes, kernel_sizes):
        super(CNNTransformer, self).__init__()
        self.cnn_encoder = CNNEncoder(conv_sizes, kernel_sizes)  # Your CNN encoder
        self.linear_projection = nn.Linear(8, transformer_d_model)  # Project to d_model
        self.transformer = nn.Transformer(
            d_model=transformer_d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )

    def forward(self, x):
        cnn_output = self.cnn_encoder(x)  # Shape: (batch_size, 8, 29)
        cnn_output = cnn_output.permute(0, 2, 1)  # Shape: (batch_size, 29, 8)
        transformer_input = self.linear_projection(cnn_output)  # Shape: (batch_size, 29, d_model)
        transformer_output = self.transformer(transformer_input)
        return transformer_output
