import torch
import tqdm 

def train_model(model, train_loader, criterion, optimizer, epochs=1, device="cuda"):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        # Wrap train_loader with tqdm for progress visualization
        loop = tqdm(enumerate(train_loader, 0), total=len(train_loader), leave=True)
        for i, data in loop:
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()


            # Forward pass
            exit_1, exit_2, exit_3 = model(inputs)  # Assume early exits are ignored during training
            loss_1 = criterion(exit_1, labels)
            loss_2 = criterion(exit_2, labels)
            loss_3 = criterion(exit_3, labels)
            loss = loss_1 + loss_2 + loss_3

            # Backward and optimize
            loss.backward()
            optimizer.step()


            # Update statistics
            running_loss += loss.item()
            if i % 200 == 199:  # Update tqdm bar with average loss every 200 mini-batches
                avg_loss = running_loss / 200
                # Update tqdm postfix information
                loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                loop.set_postfix(loss=avg_loss)
                running_loss = 0.0


def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    total_samples = len(test_loader.dataset)
    correct_predictions = [0, 0, 0, 0]  # Assuming three exits

    with torch.no_grad():  # No need to compute gradients
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            exits = model(inputs)  # Forward pass
            softmax = torch.nn.Softmax(dim=1)
            
            for i, exit in enumerate(exits):
                predictions = softmax(exit).argmax(dim=1)
                correct_predictions[i] += (predictions == labels).type(torch.float).sum().item()

    accuracies = [correct / total_samples for correct in correct_predictions]
    return accuracies
