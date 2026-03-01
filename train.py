#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("Missing dependencies for train.py. Run: pip install -r requirements.txt") from exc

from slr_baseline.data import KeypointNPZDataset, load_manifest
from slr_baseline.model import SignBiLSTMBaseline
from slr_baseline.utils import set_seed


def topk_hits(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    hits = pred.eq(targets.view(-1, 1)).any(dim=1).sum().item()
    return int(hits)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_top5 = 0

    pbar = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for batch_idx, (x, y, _) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = y.shape[0]
        total_samples += bsz
        total_loss += float(loss.item()) * bsz
        total_top1 += topk_hits(logits, y, k=1)
        total_top5 += topk_hits(logits, y, k=5)

        pbar.set_postfix(
            loss=f"{total_loss / max(total_samples,1):.4f}",
            top1=f"{total_top1 / max(total_samples,1):.4f}",
            top5=f"{total_top5 / max(total_samples,1):.4f}",
        )

        if batch_idx == 1:
            print(f"[train] first batch x={tuple(x.shape)} y={tuple(y.shape)}")

    return {
        "loss": total_loss / max(total_samples, 1),
        "top1": total_top1 / max(total_samples, 1),
        "top5": total_top5 / max(total_samples, 1),
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    min_lr: float,
):
    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_type}")


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSL Hands+Pose baseline")
    parser.add_argument("--manifest", type=str, default="meta/manifest.csv")
    parser.add_argument("--processed-root", type=str, default="dataset_processed")
    parser.add_argument("--vocab", type=str, default="meta/vocab_gloss.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--no-normalize", action="store_true", help="Disable per-sample keypoint normalization")
    parser.add_argument("--no-velocity", action="store_true", help="Disable temporal velocity features")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}")

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"[train] device={device}")

    manifest_rows = load_manifest(args.manifest)
    if not manifest_rows:
        raise RuntimeError("Manifest is empty")
    num_classes = max(r.label_id for r in manifest_rows) + 1

    split_counts = defaultdict(int)
    for r in manifest_rows:
        split_counts[r.split] += 1
    print(
        "[train] manifest split counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items()))
    )

    normalize = not args.no_normalize
    use_velocity = not args.no_velocity

    train_ds = KeypointNPZDataset(
        args.manifest,
        args.processed_root,
        split="train",
        strict=True,
        normalize=normalize,
        use_velocity=use_velocity,
    )
    val_ds = KeypointNPZDataset(
        args.manifest,
        args.processed_root,
        split="val",
        strict=True,
        normalize=normalize,
        use_velocity=use_velocity,
    )
    test_ds = KeypointNPZDataset(
        args.manifest,
        args.processed_root,
        split="test",
        strict=True,
        normalize=normalize,
        use_velocity=use_velocity,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    sample_x, _, _ = train_ds[0]
    input_dim = int(sample_x.shape[-1])
    num_frames = int(sample_x.shape[0])

    model = SignBiLSTMBaseline(
        num_classes=num_classes,
        input_dim=input_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        min_lr=args.min_lr,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    best_val_top1 = -1.0

    print(
        f"[train] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"num_classes={num_classes} input_dim={input_dim} T={num_frames} "
        f"normalize={normalize} velocity={use_velocity} "
        f"scheduler={args.scheduler} ckpt={ckpt_path.resolve()}"
    )

    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[train] epoch {epoch}/{args.epochs}")
        print(f"[train] lr={lr_now:.8f}")
        tr = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        va = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            "[train] "
            f"train loss={tr['loss']:.4f} top1={tr['top1']:.4f} top5={tr['top5']:.4f} | "
            f"val loss={va['loss']:.4f} top1={va['top1']:.4f} top5={va['top5']:.4f}"
        )
        if scheduler is not None:
            scheduler.step()

        if va["top1"] > best_val_top1:
            best_val_top1 = va["top1"]
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_top1": best_val_top1,
                "config": {
                    "num_classes": num_classes,
                    "input_dim": input_dim,
                    "proj_dim": args.proj_dim,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "num_frames": num_frames,
                    "normalize": normalize,
                    "use_velocity": use_velocity,
                },
            }
            torch.save(ckpt, ckpt_path)
            print(f"[train] saved best checkpoint: {ckpt_path} (val_top1={best_val_top1:.4f})")

    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found after training: {ckpt_path}")

    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    te = run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(
        f"\n[test] loss={te['loss']:.4f} top1={te['top1']:.4f} top5={te['top5']:.4f} "
        f"(best_val_top1={best.get('best_val_top1', -1):.4f})"
    )

    vocab_path = Path(args.vocab)
    if vocab_path.exists():
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(
            f"[train] vocab loaded: {vocab_path.resolve()} "
            f"used_label_count={vocab.get('used_label_count', 'NA')}"
        )


if __name__ == "__main__":
    main()
