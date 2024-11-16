import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        
    def forward(self, src):
        output = self.transformer_encoder(src)
        return output