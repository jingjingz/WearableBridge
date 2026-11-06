"""Train a bridge from cached features and a frozen reference classifier."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wearablebridge import MeanFlow, ResidualBridge, Standardizer, bridge_loss, predict


def _state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def validation_f1(bridge, classifier, standardizer, batches, device):
    confusion = None
    for features, labels in batches:
        if labels.dtype != torch.long or labels.shape != (len(features),):
            raise ValueError("validation labels must be aligned int64 class indices")
        probabilities = predict(features.to(device), bridge, classifier, standardizer)
        classes = probabilities.shape[1]
        labels = labels.to(device)
        if (
            not torch.isfinite(probabilities).all()
            or ((labels < 0) | (labels >= classes)).any()
        ):
            raise ValueError(
                "validation requires finite predictions and labels in range"
            )
        counts = (
            torch.bincount(
                labels * classes + probabilities.argmax(1), minlength=classes**2
            )
            .reshape(classes, classes)
            .cpu()
        )
        confusion = counts if confusion is None else confusion + counts
    if confusion is None or confusion.sum() == 0:
        raise ValueError("validation must contain labeled source features")
    denominator = confusion.sum(0) + confusion.sum(1)
    return (2 * confusion.diag() / denominator.clamp_min(1)).mean().item()


def train_bridge(
    batches,
    reference_train,
    classifier,
    *,
    method="meanflow",
    hidden_dim=256,
    epsilon=0.01,
    epochs=30,
    learning_rate=1e-4,
    weight_decay=1e-4,
    validation=None,
    min_epochs=5,
    patience=5,
    min_delta=1e-4,
    seed=0,
    device="cpu",
):
    """Fit on (source, reference, labels, weights) batches.

    batches may be a DataLoader or a function(epoch) that rematches pairs.
    validation yields (source, labels); omitting it keeps the final epoch.
    """

    if method not in ("mapper", "meanflow", "distillation"):
        raise ValueError(f"unknown method: {method}")
    if min(epochs, min_epochs, patience) < 1 or min_delta < 0 or epsilon <= 0:
        raise ValueError(
            "use positive epochs, patience and epsilon, and min_delta >= 0"
        )
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("use cpu or cuda for float32 MeanFlow training")
    torch.manual_seed(seed)
    classifier = (
        classifier.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    )
    standardizer = Standardizer(reference_train.float()).to(device)
    dim = len(standardizer.mean)
    bridge = (
        MeanFlow(dim, hidden_dim)
        if method == "meanflow"
        else ResidualBridge(dim, hidden_dim)
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    history, best_state = [], None
    best, material_best, stale, selected_epoch = -float("inf"), -float("inf"), 0, 0

    for epoch in range(1, epochs + 1):
        bridge.train()
        total, count = 0.0, 0
        epoch_batches = batches(epoch) if callable(batches) else batches
        for source, reference, labels, weights in epoch_batches:
            source, reference = source.to(device), reference.to(device)
            labels, weights = labels.to(device), weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = bridge_loss(
                method,
                bridge,
                classifier,
                standardizer,
                source,
                reference,
                labels,
                weights,
                generator=generator,
                epsilon=epsilon,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(bridge.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
        if count == 0:
            raise ValueError("each epoch must contain training pairs")
        record = {"epoch": epoch, "loss": total / count}
        if validation is not None:
            score = validation_f1(bridge, classifier, standardizer, validation, device)
            record["validation_macro_f1"] = score
            if score > best:
                best, best_state, selected_epoch = score, _state(bridge), epoch
            if score > material_best + min_delta:
                material_best, stale = score, 0
            else:
                stale += 1
        else:
            selected_epoch = epoch
        history.append(record)
        if validation is not None and epoch >= min_epochs and stale >= patience:
            break
    if best_state is not None:
        bridge.load_state_dict(best_state)
    checkpoint = {
        "method": method,
        "feature_dim": dim,
        "hidden_dim": hidden_dim,
        "bridge": _state(bridge),
        "normalization": _state(standardizer),
        "classifier": _state(classifier),
        "linear_classifier": type(classifier) is nn.Linear,
        "selected_epoch": selected_epoch,
        "history": history,
        "settings": {
            "epsilon": epsilon,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "epochs": epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "selection": "validation_macro_f1"
            if validation is not None
            else "final_epoch",
        },
    }
    return checkpoint


def load_bridge(path, *, classifier=None, device="cpu"):
    """Restore the bridge, frozen classifier, and saved normalization."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["method"] not in ("mapper", "meanflow", "distillation"):
        raise ValueError("unrecognized bridge method in checkpoint")
    dim, hidden = checkpoint["feature_dim"], checkpoint["hidden_dim"]
    bridge = (
        MeanFlow(dim, hidden)
        if checkpoint["method"] == "meanflow"
        else ResidualBridge(dim, hidden)
    )
    bridge.load_state_dict(checkpoint["bridge"])
    if classifier is None:
        if not checkpoint["linear_classifier"]:
            raise ValueError(
                "provide the original classifier architecture to load this checkpoint"
            )
        state = checkpoint["classifier"]
        classifier = nn.Linear(dim, state["weight"].shape[0], bias="bias" in state)
    classifier.load_state_dict(checkpoint["classifier"])
    standardizer = Standardizer(**checkpoint["normalization"])
    return (
        bridge.to(device=device, dtype=torch.float32).eval(),
        classifier.to(device=device, dtype=torch.float32).eval().requires_grad_(False),
        standardizer.to(device=device, dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument(
        "--method", choices=("mapper", "meanflow", "distillation"), default="meanflow"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    source, reference, labels = data["source"], data["reference"], data["labels"]
    if (
        source.dtype != torch.float32
        or reference.dtype != torch.float32
        or labels.dtype != torch.long
    ):
        raise ValueError("use float32 features and int64 labels")
    state = data["classifier"]
    classifier = nn.Linear(
        source.shape[1], state["weight"].shape[0], bias="bias" in state
    )
    classifier.load_state_dict(state)
    weights = data.get("weights", torch.ones(len(labels)))
    batches = DataLoader(
        TensorDataset(source, reference, labels, weights),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation = None
    if args.validation:
        val = torch.load(args.validation, map_location="cpu", weights_only=True)
        validation = DataLoader(
            TensorDataset(val["source"], val["labels"]), batch_size=args.batch_size
        )
    checkpoint = train_bridge(
        batches,
        data["reference_train"],
        classifier,
        method=args.method,
        hidden_dim=args.hidden_dim,
        epsilon=args.epsilon,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation=validation,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    for row in checkpoint["history"]:
        line = f"epoch {row['epoch']:02d}: loss={row['loss']:.6f}"
        if "validation_macro_f1" in row:
            line += f", validation macro-F1={row['validation_macro_f1']:.4f}"
        print(line)
    print(f"Saved epoch {checkpoint['selected_epoch']} to {args.output}")


if __name__ == "__main__":
    main()
