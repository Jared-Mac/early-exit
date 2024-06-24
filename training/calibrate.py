import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0, requires_grad=True, device='cpu'):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1, device=device) * temperature).to(device)
        if not requires_grad:
            self.temperature.requires_grad = False

    def forward(self, logits):
        # No change needed here, as self.temperature is already on the correct device
        return logits / self.temperature

def find_optimal_temperature(logits, labels,device):
    """
    Find the temperature that minimizes the NLL loss.
    """
    # Initialize the temperature model
    temp_model = TemperatureScaling(temperature=1.0, device=device)
    
    # Define optimizer for temperature parameter
    optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(temp_model(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temp_model.temperature.item()
def calibrate_model_exits(model, valid_loader, device):
    model.eval()  # Set the model to evaluation mode
    logits_list_exit1 = []
    logits_list_exit2 = []
    logits_list_final = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output_early_exit1, output_early_exit2, output_final = model(inputs)
            logits_list_exit1.append(output_early_exit1)
            logits_list_exit2.append(output_early_exit2)
            logits_list_final.append(output_final)
            labels_list.append(labels)

    # Concatenate lists of logits and labels
    logits_exit1 = torch.cat(logits_list_exit1)
    logits_exit2 = torch.cat(logits_list_exit2)
    logits_final = torch.cat(logits_list_final)
    labels = torch.cat(labels_list)

    # Find the optimal temperature for each exit
    optimal_temperature_exit1 = find_optimal_temperature(logits_exit1, labels,device)
    optimal_temperature_exit2 = find_optimal_temperature(logits_exit2, labels,device)
    optimal_temperature_final = find_optimal_temperature(logits_final, labels,device)

    return optimal_temperature_exit1, optimal_temperature_exit2, optimal_temperature_final

def calibrate_model(model, valid_loader, device):
    model.eval()  # Set the model to evaluation mode
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)[-1]  # Use the final exit for calibration
            logits_list.append(outputs)
            labels_list.append(labels)
            
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Find the optimal temperature
    optimal_temperature = find_optimal_temperature(logits, labels,device)
    print(f"Optimal temperature: {optimal_temperature}")
    return optimal_temperature

def inference_with_temperature_scaling(model, inputs, temperature):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)[-1]  # Assuming you're using the final exit for predictions
        scaled_outputs = outputs / temperature
        probabilities = F.softmax(scaled_outputs, dim=1)
    return probabilities

