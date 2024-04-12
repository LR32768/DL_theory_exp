import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_change(initial_parameters, new_parameters):
    with torch.no_grad():
        total_change = 0.0
        total_initial_norm = 0.0
        for initial, new in zip(initial_parameters, new_parameters):
            total_change += torch.norm(new - initial).item()
            total_initial_norm += torch.norm(initial).item()
        return total_change / total_initial_norm if total_initial_norm > 0 else 0


def IsSimilar(x, y):
    if abs(x - y) < 1e-5:
        return 1
    else:
        return 0


batch_size = 1024

dataset = torchvision.datasets.MNIST('~/data/', train=True, download=True,
                                     transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST('~/data/', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())
# Only extract one batch of data for faster training

# Adjust learning rate and increase training steps
learning_rate = 1e-2
widths = [10, 100, 1000, 10000, 50000, 100000]

results = []

for width in widths:
    print("Width: ", width)
    # Adjust learning rate inversely with width
    scaled_learning_rate = learning_rate / np.sqrt(width)
    epochs = 500
    model = nn.Sequential(
        nn.Linear(28 * 28, width),
        nn.ReLU(),
        nn.Linear(width, 10)
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=scaled_learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    initial_parameters = [p.clone() for p in model.parameters()]

    final_loss = 0.0

    losses = []
    acc = 0

    # tqdm updating the loss

    with tqdm(total=epochs, unit='batch') as pbar:
        for ep in range(epochs):
            model.train()
            batch_X, batch_y = dataset.data[:batch_size], dataset.targets[:batch_size]
            optimizer.zero_grad()
            batch_X = batch_X.view(-1, 28 * 28).to(device).float()
            output = model(batch_X)
            loss = loss_fn(output, batch_y.to(device))
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            losses.append(final_loss)
            pbar.set_description(f'epoch: {ep + 1}')
            pbar.set_postfix({'loss': final_loss})
            pbar.update(1)

    with torch.no_grad():
        total, correct = 0, 0
        model.eval()
        x, y = test_dataset.data, test_dataset.targets
        x = x.view(-1, 28 * 28).to(device).float()
        output = model(x)
        _, predicted = torch.max(output.data, dim=1)
        total += y.size(0)
        correct += (predicted.cpu() == y).sum().item()
        acc = correct / total
    print("acc: ", acc * 100)
    new_parameters = [p for p in model.parameters()]
    change = relative_change(initial_parameters, new_parameters)
    results.append((width, change, final_loss, acc))
    # torch.save(model, str(width) + '.pth')

    plt.plot(range(len(losses)), losses)
    plt.ylabel("train_loss")
    plt.xlabel("epoch")
    plt.savefig("curve_" + str(width) + ".png")
    plt.clf()
    print(results)

print(results)
with open('result.txt', 'w') as f:
    for item in results:
        f.write(str(item) + '\n')