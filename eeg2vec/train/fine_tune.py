import torch
from torch import nn, optim
from eeg2vec.data_loader import get_dataloader  # Adjust according to your dataset
from models.eeg_model import EEGModel  # Your self-supervised model
from loss.contrastive_loss import ContrastiveLoss  # Optional, if you want to keep contrastive loss
from utils.utils import evaluate  # Function for model evaluation

def fine_tune_model(data, labels, model, num_epochs=10, batch_size=32, learning_rate=1e-3):
    # Create data loader for the fine-tuning dataset
    dataloader = get_dataloader(data, labels, batch_size)

    # Set model to training mode
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Change as per your classification task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Optional: Evaluate the model on a validation set
        evaluate(model, validation_data, validation_labels)

    return model

if __name__ == "__main__":
    # Load your pre-trained model
    model = EEGModel()  # Load the model architecture
    model.load_state_dict(torch.load('path_to_your_pretrained_model.pth'))  # Load the pre-trained weights

    # Load your fine-tuning data (EEG data with labels)
    fine_tune_data = ...  # Your EEG fine-tuning data
    fine_tune_labels = ...  # Corresponding labels

    # Fine-tune the model
    fine_tuned_model = fine_tune_model(fine_tune_data, fine_tune_labels, model)
