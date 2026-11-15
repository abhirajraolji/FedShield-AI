print("Federated DP file started!", flush=True)
from opacus.accountants import RDPAccountant    
import copy
import json
import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from src.dataset import load_dataset, create_clients
from src.model import MedicalCNN
from src.federated import fedavg, evaluate, DEVICE


ROUNDS = 3
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
NOISE_MULTIPLIER = 0.8
MAX_GRAD_NORM = 1.0
DELTA = 1e-5


def train_private_client(global_model, dataset, accountant):
    # Each hospital receives its own copy of the global model
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    privacy_engine = PrivacyEngine(accountant='rdp')
    privacy_engine.accountant = accountant

    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(LOCAL_EPOCHS):
        model.train()

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.view(-1).long().to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    epsilon = accountant.get_epsilon(DELTA)

    # Opacus wraps the original model.
    # Extract its ordinary PyTorch state dictionary.
    weights = {
        key: value.detach().cpu().clone()
        for key, value in model._module.state_dict().items()
    }

    return weights, float(epsilon)


def run_private_federated():
    torch.manual_seed(42)

    train_data, test_data = load_dataset()

    clients = create_clients(
        train_data,
        num_clients=4,
        samples=8000,
        alpha=0.5
    )
    accountants = [
    RDPAccountant() for _ in clients
]

    global_model = MedicalCNN().to(DEVICE)

    history = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n=== PRIVATE ROUND {round_num} ===")

        client_weights = []
        client_sizes = []
        client_epsilons = []

        for i, client in enumerate(clients):
            print(f"Training Hospital {i + 1} with DP...")

            weights, epsilon = train_private_client(
                global_model,
                client,
                accountants[i]
            )

            client_weights.append(weights)
            client_sizes.append(len(client))
            client_epsilons.append(epsilon)

            print(
                f"Hospital {i + 1}: "
                f"epsilon = {epsilon:.3f}"
            )

        new_weights = fedavg(
            client_weights,
            client_sizes
        )

        global_model.load_state_dict(new_weights)

        accuracy = evaluate(
            global_model,
            test_data
        )

        print(f"Global Accuracy: {accuracy:.2f}%")

        history.append({
            "round": round_num,
            "accuracy": accuracy,
            "cumulative_client_epsilons": client_epsilons
        })

    os.makedirs("results", exist_ok=True)

    with open(
        "results/federated_dp.json", "w"
    ) as f:
        json.dump(history, f, indent=2)

    torch.save(
        global_model.state_dict(),
        "results/private_global_model.pth"
    )

    print("\nPrivate federated training complete!")

if __name__ == "__main__":
    print("Starting private federated training...", flush=True)
    run_private_federated()