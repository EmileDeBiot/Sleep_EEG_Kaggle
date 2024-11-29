import torch
import torch.nn as nn
import torch.optim as optim

def train_cnn_classifier(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    """
    Parameters:
        model: PyTorch model
        dataloader: DataLoader for training data
        optimizer: Optimizer (e.g., Adam)
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    """
    # Loss function: Binary Cross-Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    # Move model to device (e.g., GPU)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for inputs, labels in dataloader:
            # Zero the gradient buffers
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)

            # Ensure data types are compatible
            inputs = inputs.float()   # Convert inputs to float32
            labels = labels.float()

            
            # Forward pass
            outputs = model(inputs)  # Outputs: (batch_size, 5)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Epoch stats
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    print("Training complete.")

# Example usage
# Assuming model, dataloader, and optimizer are already defined
# train_model(model, dataloader, optimizer, num_epochs=20, device='cuda')
