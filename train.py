'''from src.federated import run_federated

if __name__ == "__main__":
    print("Starting FedShield AI...")
    run_federated(rounds=3)'''


import copy
import json
import os

import torch
from torch.utils.data import ConcatDataset

from src.dataset import load_dataset, create_clients
from src.model import MedicalCNN
from src.federated import (
    train_local,
    fedavg,
    evaluate,
    DEVICE
)

ROUNDS = 3
SEED = 42


def run_comparison():
    print("Starting FedShield AI Comparison...")

    torch.manual_seed(SEED)

    train_data, test_data = load_dataset()

    clients = create_clients(
        train_data,
        num_clients=4,
        samples=8000,
        alpha=0.5
    )

    # Same initial weights for all methods
    initial_model = MedicalCNN().to(DEVICE)
    initial_weights = copy.deepcopy(
        initial_model.state_dict()
    )

    results = {
        "local": [],
        "federated": [],
        "centralized": None,
        "federated_history": []
    }

    # EXPERIMENT 1: LOCAL-ONLY
    print("\n=== LOCAL TRAINING ===")

    for i, client in enumerate(clients):
        model = MedicalCNN().to(DEVICE)
        model.load_state_dict(initial_weights)

        weights = train_local(
            model, client, epochs=ROUNDS
        )

        model.load_state_dict(weights)

        accuracy = evaluate(model, test_data)

        results["local"].append(accuracy)

        print(
            f"Hospital {i+1}: "
            f"{accuracy:.2f}%"
        )

    # EXPERIMENT 2: FEDERATED
    print("\n=== FEDERATED TRAINING ===")

    global_model = MedicalCNN().to(DEVICE)
    global_model.load_state_dict(initial_weights)

    for round_num in range(1, ROUNDS + 1):
        client_weights = []
        client_sizes = []

        for client in clients:
            if len(client) == 0:
                continue

            weights = train_local(
                global_model, client, epochs=1
            )

            client_weights.append(weights)
            client_sizes.append(len(client))

        aggregated = fedavg(
            client_weights, client_sizes
        )

        global_model.load_state_dict(aggregated)

        accuracy = evaluate(
            global_model, test_data
        )

        results["federated_history"].append({
            "round": round_num,
            "accuracy": accuracy
        })

        print(
            f"Round {round_num}: "
            f"{accuracy:.2f}%"
        )

    results["federated"] = accuracy

    # EXPERIMENT 3: CENTRALIZED
    print("\n=== CENTRALIZED TRAINING ===")

    pooled_data = ConcatDataset(clients)

    centralized_model = MedicalCNN().to(DEVICE)
    centralized_model.load_state_dict(
        initial_weights
    )

    weights = train_local(
        centralized_model,
        pooled_data,
        epochs=ROUNDS
    )

    centralized_model.load_state_dict(weights)

    central_accuracy = evaluate(
        centralized_model, test_data
    )

    results["centralized"] = central_accuracy

    print(
        f"Centralized Accuracy: "
        f"{central_accuracy:.2f}%"
    )

    # SAVE RESULTS
    os.makedirs("results", exist_ok=True)

    with open(
        "results/comparison.json", "w"
    ) as f:
        json.dump(results, f, indent=2)

    torch.save(
        global_model.state_dict(),
        "results/global_model.pth"
    )

    print("\n=== FINAL RESULTS ===")

    for i, acc in enumerate(results["local"]):
        print(
            f"Hospital {i+1}: {acc:.2f}%"
        )

    print(
        f"Federated: "
        f"{results['federated']:.2f}%"
    )

    print(
        f"Centralized: "
        f"{results['centralized']:.2f}%"
    )

    print("\nComparison saved successfully!")


if __name__ == "__main__":
    run_comparison()

