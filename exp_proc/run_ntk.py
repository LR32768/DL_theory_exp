import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def relative_change(initial_parameters, new_parameters):
    with torch.no_grad():
        total_change = 0.0
        total_initial_norm = 0.0
        for initial, new in zip(initial_parameters, new_parameters):
            total_change += torch.norm(new - initial).item()
            total_initial_norm += torch.norm(initial).item()
        return total_change / total_initial_norm if total_initial_norm > 0 else 0


# Adjusted experiment setup
X = torch.linspace(-3, 3, steps=100).unsqueeze(1).cuda()  # 100 samples, single feature in range [-3, 3]
y = torch.sin(X)  # Simpler target function

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Adjust learning rate and increase training steps
learning_rate = 0.001
epochs = 500  # Increase number of training epochs
widths = [10, 100, 1000, 10000]

results = []

for width in widths:
    # Adjust learning rate inversely with width
    scaled_learning_rate = learning_rate / width
    model = nn.Sequential(
        nn.Linear(1, width),
        nn.ReLU(),
        nn.Linear(width, 1)
    ).cuda()

    optimizer = optim.SGD(model.parameters(), lr=scaled_learning_rate)
    loss_fn = nn.MSELoss()

    initial_parameters = [p.clone() for p in model.parameters()]

    final_loss = 0.0
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
        final_loss = loss.item()

    new_parameters = [p for p in model.parameters()]
    change = relative_change(initial_parameters, new_parameters)
    results.append((width, change, final_loss))


print(results)
