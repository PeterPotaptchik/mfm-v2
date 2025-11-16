import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # 32x28x28
        x = F.relu(self.conv2(x))       # 64x28x28
        x = self.pool(x)                # 64x14x14
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                 # logits
        return x


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
    return total_loss / total, correct / total  # (avg CE), accuracy


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(outdir, model, optimizer, epoch, best_acc, tag="last"):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "best_acc": best_acc,
        "torch_version": torch.__version__,
    }
    path = Path(outdir) / f"checkpoint_{tag}.pt"
    torch.save(ckpt, path)
    return str(path)

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("best_acc", 0.0)
    return start_epoch, best_acc


def plot_metrics(epochs, test_losses, test_accs, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure()
    plt.plot(epochs, test_accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Test Accuracy")
    acc_path = outdir / "test_accuracy.png"
    plt.grid(True)
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    # Cross-entropy
    plt.figure()
    plt.plot(epochs, test_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test Cross-Entropy")
    plt.title("MNIST Test Cross-Entropy")
    loss_path = outdir / "test_cross_entropy.png"
    plt.grid(True)
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()
    return str(acc_path), str(loss_path)

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),                      # [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
    ])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    # Model / Optim
    model = MNISTCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_acc = 0.0

    # Training loop
    test_acc_hist = []
    test_loss_hist = []
    epoch_indices = []

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device,)
        test_loss, test_acc = evaluate(model, test_loader, device)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        epoch_indices.append(epoch + 1)

        print(f"Epoch {epoch+1}/{args.epochs} "
              f"| train CE: {train_loss:.4f} acc: {train_acc:.4f} "
              f"| test CE: {test_loss:.4f} acc: {test_acc:.4f}")

        last_path = save_checkpoint(outdir, model, optimizer, epoch, best_acc, tag="last")
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = save_checkpoint(outdir, model, optimizer, epoch, best_acc, tag="best")
            print(f"[ckpt] New best acc {best_acc:.4f}. Saved: {best_path}")
        else:
            print(f"[ckpt] Saved last checkpoint: {last_path}")

    metrics = {
        "epochs": epoch_indices,
        "test_cross_entropy": test_loss_hist,
        "test_accuracy": test_acc_hist,
        "best_test_accuracy": best_acc,
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    acc_fig, loss_fig = plot_metrics(epoch_indices, test_loss_hist, test_acc_hist, outdir)
    print(f"[done] Best test accuracy: {best_acc:.4f}")
    print(f"[done] Plots saved to:\n  - {acc_fig}\n  - {loss_fig}")
    print(f"[done] Metrics JSON: {outdir / 'metrics.json'}")
    print(f"[done] Checkpoints in: {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple CNN on MNIST (PyTorch).")
    parser.add_argument("--data-dir", type=str, default="/vols/bitbucket/saravanan/distributional-mf/data")
    parser.add_argument("--outdir", type=str, default="/vols/bitbucket/saravanan/distributional-mf/classifiers/mnist",)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)