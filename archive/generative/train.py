# Train VAE model

import torch
import torch.nn as nn
import torch.optim as optim
from generative.vae import VAE

def train_vae(data, model, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x in data:
            x = x.to("cuda")
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss = model.loss(x, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Reconstruction quality
        total_loss /= len(data)
        print('Epoch %d, Loss: %.4f' % (epoch, total_loss))