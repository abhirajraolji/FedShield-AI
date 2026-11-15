
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.model import MedicalCNN
from src.dataset import load_dataset, create_clients


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


def train_local(global_model, dataset, epochs=2):
    model = copy.deepcopy(global_model).to(DEVICE)

    loader = DataLoader(
        dataset, batch_size=64, shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.view(-1).long().to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def fedavg(client_weights, client_sizes):
    total = sum(client_sizes)
    aggregated = {}

    for key in client_weights[0]:
        aggregated[key] = sum(
            weights[key] * (size / total)
            for weights, size in zip(
                client_weights, client_sizes
            )
        )

    return aggregated


def evaluate(model, dataset):
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False
    )

    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.view(-1).long().to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total


def run_federated(rounds=10):
    train_data, test_data = load_dataset()

    clients = create_clients(
        train_data, num_clients=4,
        samples=8000, alpha=0.5
    )

    global_model = MedicalCNN().to(DEVICE)

    history = []

    for round_num in range(1, rounds + 1):
        print(f"\n--- ROUND {round_num} ---")

        client_weights = []
        client_sizes = []

        for i, client in enumerate(clients):
            if len(client) == 0:
                continue

            print(f"Training Hospital {i+1}...")

            weights = train_local(
                global_model, client
            )

            client_weights.append(weights)
            client_sizes.append(len(client))

        new_weights = fedavg(
            client_weights, client_sizes
        )

        global_model.load_state_dict(new_weights)

        accuracy = evaluate(
            global_model, test_data
        )

        print(f"Global Accuracy: {accuracy:.2f}%")

        history.append({
            "round": round_num,
            "accuracy": accuracy,
            "clients": len(client_weights)
        })

    os.makedirs("results", exist_ok=True)

    with open("results/history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        global_model.state_dict(),
        "results/global_model.pth"
    )

    print("\nTraining complete!")
    print("Results saved in results/")

    return history


if __name__ == "__main__":
    run_federated(rounds=10)
