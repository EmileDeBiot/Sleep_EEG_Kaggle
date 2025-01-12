import torch
from torch import optim
from eeg2vec.models.eeg2vec import EEG2Vec
from eeg2vec.contrastive_loss import ContrastiveLoss

def mask_input(input_data, mask_prob=0.20):
    masked_data = input_data.clone()
    mask = torch.rand(masked_data.size()) < mask_prob
    masked_data[mask] = 0
    return masked_data

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    for batch in dataloader:
        inputs, labels = batch
        # convert to float
        inputs = inputs.float()
        # move to device
        inputs = inputs.to(device)
        masked_inputs = mask_input(inputs)
        
        output1 = model(inputs)
        output2 = model(masked_inputs)

        loss = loss_fn(output1, output2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train(model, dataloader, num_epochs, device, lr=1e-3):
    loss_fn = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        train_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")