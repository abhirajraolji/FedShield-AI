
import copy
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import load_dataset, create_clients
from src.model import MedicalCNN
from src.federated import fedavg, evaluate, DEVICE


ROUNDS = 10
LOCAL_EPOCHS = 1
MU = 0.01
LEARNING_RATE = 0.001


def train_fedprox(global_model, dataset):
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # Save the original global parameters
    global_parameters = [
        p.detach().clone()
        for p in global_model.parameters()
    ]

    for epoch in range(LOCAL_EPOCHS):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.view(-1).long().to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            classification_loss = criterion(
                outputs, labels
            )

            # FedProx penalty
            proximal_term = torch.zeros(
                (), device=DEVICE
            )

            for local_p, global_p in zip(
                model.parameters(),
                global_parameters
            ):
                proximal_term += torch.sum(
                    (local_p - global_p) ** 2
                )

            loss = (
                classification_loss
                + (MU / 2) * proximal_term
            )

            loss.backward()
            optimizer.step()

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def run_fedprox():
    torch.manual_seed(42)

    train_data, test_data = load_dataset()

    clients = create_clients(
        train_data,
        num_clients=4,
        samples=8000,
        alpha=0.5
    )

    model = MedicalCNN().to(DEVICE)

    history = []

    for round_num in range(1, ROUNDS + 1):
        print(
            f"\n=== FedProx Round {round_num} ===",
            flush=True
        )

        client_weights = []
        client_sizes = []

        for i, client in enumerate(clients):
            print(
                f"Training Hospital {i + 1}...",
                flush=True
            )

            weights = train_fedprox(
                model,
                client
            )

            client_weights.append(weights)
            client_sizes.append(len(client))

        aggregated_weights = fedavg(
            client_weights,
            client_sizes
        )

        model.load_state_dict(aggregated_weights)

        accuracy = evaluate(model, test_data)

        print(
            f"Round {round_num}: {accuracy:.2f}%",
            flush=True
        )

        history.append({
            "round": round_num,
            "accuracy": accuracy
        })

    os.makedirs("results", exist_ok=True)

    with open(
        "results/fedprox.json", "w"
    ) as f:
        json.dump(history, f, indent=2)

    print("\nFedProx experiment complete!")


if __name__ == "__main__":
    run_fedprox()
