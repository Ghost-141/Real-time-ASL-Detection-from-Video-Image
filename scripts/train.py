import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dataset + augmentation
# -------------------------
class NpyDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        seed: int = 42,
        include_bones: bool = True,
        include_angles: bool = True,
        include_hand_present: bool = True,
        mirror_prob: float = 0.5,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Must match your extractor/preprocess.json
        self.include_bones = include_bones
        self.include_angles = include_angles
        self.include_hand_present = include_hand_present
        self.mirror_prob = float(mirror_prob)

        # Feature layout (from our extractor)
        self.lm_dim = 63  # 21*3
        self.bone_dim = 60 if include_bones else 0  # 20*3
        self.angle_dim = 13 if include_angles else 0
        self.hp_dim = 1 if include_hand_present else 0

        expected_min = self.lm_dim + self.bone_dim + self.angle_dim + self.hp_dim
        if self.X.shape[1] < expected_min:
            raise ValueError(
                f"X has dim {self.X.shape[1]} but expected at least {expected_min}. "
                "Check include_bones/include_angles/include_hand_present."
            )

    def __len__(self):
        return self.X.shape[0]

    def _mirror_inplace(self, x: np.ndarray):
        """
        Mirrors the feature vector in-place across the X axis.
        Only applies to:
          - landmarks (first 63 dims -> (21,3), mirror x channel)
          - bones (next 60 dims -> (20,3), mirror x channel)
        Angles are reflection-invariant, and hand_present should not change.
        """
        # Mirror landmarks
        lm = x[: self.lm_dim].reshape(21, 3)
        lm[:, 0] *= -1.0
        x[: self.lm_dim] = lm.reshape(-1)

        # Mirror bone vectors (if present)
        if self.include_bones:
            start = self.lm_dim
            end = start + self.bone_dim
            bones = x[start:end].reshape(20, 3)
            bones[:, 0] *= -1.0
            x[start:end] = bones.reshape(-1)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.augment:
            # 0) Mirror augmentation (fixes left/right + webcam mirroring)
            if self.rng.random() < self.mirror_prob:
                self._mirror_inplace(x)

            # 1) Small gaussian noise
            if self.rng.random() < 0.8:
                x += self.rng.normal(0, 0.01, size=x.shape).astype(np.float32)

            # 2) Random scaling
            if self.rng.random() < 0.6:
                scale = self.rng.uniform(0.95, 1.05)
                x *= np.float32(scale)

            # 3) Random feature dropout (simulate occlusion/crop)
            if self.rng.random() < 0.3:
                m = self.rng.random(x.shape) < 0.02
                x[m] = 0.0

        return torch.from_numpy(x), torch.tensor(y)


# -------------------------
# Models
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Train
# -------------------------
def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", type=str, default="prepared")
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="weights")
    args = parser.parse_args()

    X_train = np.load(os.path.join(args.prepared_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.prepared_dir, "y_train.npy"))

    with open(os.path.join(args.prepared_dir, "labels.json"), "r") as f:
        labels = json.load(f)

    num_classes = len(labels)
    in_dim = X_train.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Train: {X_train.shape}, Classes: {num_classes}")

    ds_tr = NpyDataset(
        X_train,
        y_train,
        augment=True,
        include_bones=True,
        include_angles=True,
        mirror_prob=0.5,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    if args.model == "mlp":
        model = MLP(in_dim=in_dim, num_classes=num_classes)
    else:
        print("Model not found")

    model.to(device)

    class_w = compute_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    best_loss = float("inf")
    best_path = os.path.join(args.out_dir, "asl_classifier.pt")
    bad = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, dl_tr, device, criterion, optimizer)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f}")

        if tr_loss < best_loss - 1e-4:
            best_loss = tr_loss
            bad = 0
            model.eval()
            example_input = torch.randn(1, in_dim, device=device)
            scripted_model = torch.jit.trace(model, example_input)
            scripted_model.save(best_path)
            print(
                f"Saved best serialized model -> {best_path} (train loss {best_loss:.4f})"
            )
            model.train()
        else:
            bad += 1
            if bad >= args.patience:
                print(
                    f"Early stopping (patience={args.patience}). Best train loss: {best_loss:.4f}"
                )
                break


if __name__ == "__main__":
    main()
