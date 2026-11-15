import copy
import json
import os

import torch

from src.model import MedicalCNN
from src.dataset import load_dataset, create_clients
from src.federated import (
    train_local,
    fedavg,
    evaluate,
    DEVICE
)


ROUNDS = 3
ATTACKER_ID = 3
ATTACK_SCALE = -5.0


def poison_weights(local_weights, global_weights):
    """
    Simulate a malicious hospital that sends
    a deliberately manipulated model update.
    """
    return {
        key: global_weights[key] + ATTACK_SCALE * (
            local_weights[key] - global_weights[key]
        )
        for key in local_weights
    }


def coordinate_median(client_weights):
    """
    For each model parameter, select the median
    value across hospitals instead of the mean.
    """
    return {
        key: torch.median(
            torch.stack([w[key] for w in client_weights]),
            dim=0
        ).values
        for key in client_weights[0]
    }


def run_scenario(
    name,
    clients,
    test_data,
    initial_weights,
    attack=False,
    defence=False
):
    # Reset randomness for a fairer comparison
    torch.manual_seed(42)

    model = MedicalCNN().to(DEVICE)
    model.load_state_dict(copy.deepcopy(initial_weights))

    history = []

    print(f"\n=== {name} ===", flush=True)

    for round_num in range(1, ROUNDS + 1):
        global_weights = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        client_weights = []
        client_sizes = []

        for i, client in enumerate(clients):
            print(
                f"Training Hospital {i + 1}...",
                flush=True
            )

            weights = train_local(model, client)

            if attack and i == ATTACKER_ID:
                print(
                    f"Hospital {i + 1} sending poisoned update!",
                    flush=True
                )

                weights = poison_weights(
                    weights,
                    global_weights
                )

            client_weights.append(weights)
            client_sizes.append(len(client))

        if defence:
            new_weights = coordinate_median(client_weights)
        else:
            new_weights = fedavg(
                client_weights,
                client_sizes
            )

        model.load_state_dict(new_weights)

        accuracy = evaluate(model, test_data)

        print(
            f"Round {round_num}: {accuracy:.2f}%",
            flush=True
        )

        history.append({
            "round": round_num,
            "accuracy": accuracy
        })

    return history


def run_security_experiment():
    torch.manual_seed(42)

    train_data, test_data = load_dataset()

    clients = create_clients(
        train_data,
        num_clients=4,
        samples=8000,
        alpha=0.5
    )

    initial_weights = copy.deepcopy(
        MedicalCNN().state_dict()
    )

    results = {}

    results["normal"] = run_scenario(
        "Normal FedAvg",
        clients,
        test_data,
        initial_weights
    )

    results["attacked"] = run_scenario(
        "Attacked FedAvg",
        clients,
        test_data,
        initial_weights,
        attack=True
    )

    results["defended"] = run_scenario(
        "Attacked + Median Defence",
        clients,
        test_data,
        initial_weights,
        attack=True,
        defence=True
    )

    os.makedirs("results", exist_ok=True)

    with open("results/security.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSecurity experiment complete!", flush=True)


if __name__ == "__main__":
    run_security_experiment()
