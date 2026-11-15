'''import medmnist
import numpy as np
from medmnist import INFO

DATA_FLAG = "pathmnist"

info = INFO[DATA_FLAG]
DataClass = getattr(medmnist, info["python_class"])

def load_dataset():
    print("Downloading / loading PathMNIST...")

    train_dataset = DataClass(
        split="train",
        download=True,
        size=28
    )

    test_dataset = DataClass(
        split="test",
        download=True,
        size=28
    )

    print("Training images:", len(train_dataset))
    print("Testing images:", len(test_dataset))
    print("Number of classes:", len(info["label"]))

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_data, test_data = load_dataset()

    image, label = train_data[0]

    print("Image dimensions:", np.array(image).shape)
    print("First image label:", label)'''


import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import Subset
from torchvision import transforms

DATA_FLAG = "pathmnist"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def load_dataset():
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(
        split="train",
        transform=transform,
        download=True,
        size=28
    )

    test_dataset = DataClass(
        split="test",
        transform=transform,
        download=True,
        size=28
    )

    return train_dataset, test_dataset


def create_clients(dataset, num_clients=4,
                   samples=8000, alpha=0.5):
    rng = np.random.default_rng(42)

    labels = np.asarray(dataset.labels).flatten()
    samples = min(samples, len(labels))

    # Select a reproducible subset
    selected = rng.choice(
        len(labels), samples, replace=False
    )

    client_indices = [[] for _ in range(num_clients)]

    # Dirichlet non-IID split
    for label in np.unique(labels):
        class_indices = selected[
            labels[selected] == label
        ]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            [alpha] * num_clients
        )

        split_points = (
            np.cumsum(proportions)[:-1]
            * len(class_indices)
        ).astype(int)

        splits = np.split(class_indices, split_points)

        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    clients = [
        Subset(dataset, indices)
        for indices in client_indices
    ]

    for i, client in enumerate(clients):
        counts = np.bincount(
            labels[client.indices], minlength=9
        )
        print(
            f"Hospital {i+1}: {len(client)} images "
            f"| Class counts: {counts.tolist()}"
        )

    return clients


if __name__ == "__main__":
    train_data, test_data = load_dataset()

    print("Training samples:", len(train_data))
    print("Test samples:", len(test_data))

    clients = create_clients(train_data)

