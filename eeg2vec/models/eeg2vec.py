from eeg2vec.models.cnn_encoder import CNNEncoder
from eeg2vec.models.transformer_encoder import TransformerEncoder
import torch
from torch import nn

# Build a self-supervised learning model to learn EEG representations
# from the raw EEG sleep data. The model is trained on a large dataset of EEG
# recordings and then used to generate EEG embeddings for a smaller
# dataset of EEG recordings. The embeddings are then used to train a
# classifier to predict quality of EEG data in sleep recordings: artifact detection.

# The model is trained using a contrastive loss function, which compares
# the similarity of embeddings from the same recording to embeddings from
# different recordings.

class EEG2Vec(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(EEG2Vec, self).__init__()
        self.cnn_encoder = CNNEncoder()
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)

        return x
