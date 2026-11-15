from torch.utils.data import DataLoader, ConcatDataset
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from src.dataset import load_dataset, create_clients
from src.model import MedicalCNN
from src.federated import DEVICE


SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DELTA = 1e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_detailed(model, test_data):
    model.eval()

    loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False
    )

    correct = 0
    total = 0

    predictions_per_class = torch.zeros(
        9, dtype=torch.long
    )

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

            predictions_per_class += torch.bincount(
                predictions.cpu(),
                minlength=9
            )

    accuracy = 100 * correct / total

    return accuracy, predictions_per_class.tolist()


def run_experiment(
    name,
    train_data,
    test_data,
    noise=None,
    max_grad_norm=1.0,
    clipping_only=False
):
    print(f"\n=== {name} ===")

    set_seed(SEED)

    model = MedicalCNN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    print(f"\nDEBUG: Experiment = {name}")
    print(f"DEBUG: Training samples = {len(train_data)}")
    print(f"DEBUG: Batch size = {BATCH_SIZE}")

    loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    privacy_engine = None

    if noise is not None:
        privacy_engine = PrivacyEngine()

        model, optimizer, loader = (
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=noise,
                max_grad_norm=max_grad_norm,
                poisson_sampling=True
            )
        )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        batches = 0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.view(-1).long().to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            if clipping_only:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                )

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {total_loss/max(batches,1):.4f}"
        )

    accuracy, prediction_counts = (
        evaluate_detailed(model, test_data)
    )

    epsilon = None

    if privacy_engine is not None and noise > 0:
        epsilon = float(
        privacy_engine.get_epsilon(DELTA)
    )

    print(f"Accuracy: {accuracy:.2f}%")
    print(
        "Predictions per class:",
        prediction_counts
    )

    if epsilon is not None:
        print(f"Epsilon: {epsilon:.3f}")
        print(f"Delta: {DELTA}")

    return {
        "experiment": name,
        "accuracy": accuracy,
        "noise_multiplier": noise,
        "max_grad_norm": max_grad_norm,
        "epsilon": epsilon,
        "delta": DELTA if epsilon is not None else None,
        "predictions_per_class": prediction_counts
    }



def main():
    set_seed(SEED)

    # Step 1: Load the dataset
    train_data, test_data = load_dataset()

    # Step 2: Divide the training data among 4 hospitals
    clients = create_clients(
        train_data,
        num_clients=4,
        samples=8000,
        alpha=0.5
    )

    # Step 3: Combine all 4 hospitals for this diagnostic
    pooled_data = ConcatDataset(clients)

    print("Pooled dataset size:", len(pooled_data))
    assert len(pooled_data) == 8000

    # Keep your existing experiments list below this

    experiments = [
        {
            "name": "Pooled No Privacy",
            "noise": None,
            "clipping_only": False
        },
        {
            "name": "Pooled Opacus No Noise",
            "noise": 0.0,
            "max_grad_norm": 1.0
        },
        {
            "name": "Pooled DP Noise 0.8",
            "noise": 0.8,
            "max_grad_norm": 1.0
        }
    ]

    results = []

    for config in experiments:
        result = run_experiment(
            train_data=pooled_data,
            test_data=test_data,
            **config
        )

        results.append(result)

    os.makedirs("results", exist_ok=True)

    with open(
        "results/privacy_diagnostics.json", "w"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nDiagnostics complete!")



if __name__ == "__main__":
    main()
    

    

    assert len(pooled_data) == 8000

    results = []

    for config in experiments:
        result = run_experiment(
            train_data=hospital_data,
            test_data=test_data,
            **config
        )

        results.append(result)

    os.makedirs("results", exist_ok=True)

    with open(
        "results/privacy_diagnostics.json", "w"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nDiagnostics complete!")


if __name__ == "__main__":
    main()
