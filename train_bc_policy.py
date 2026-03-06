#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from controllers.bc_common import FEATURE_DIM, build_policy_features
DATA_DIR = Path("./data/SYNTHETIC")
OUTPUT_PATH = Path("./checkpoints/bc_policy.npz")


class BCPolicyNet(nn.Module):
  def __init__(self, in_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Tanh(),
    )

  def forward(self, x):
    return 2.0 * self.net(x).squeeze(-1)


def extract_examples(csv_path):
  df = pd.read_csv(csv_path, usecols=[
    "roll",
    "vEgo",
    "aEgo",
    "targetLateralAcceleration",
    "steerCommand",
  ])
  roll_lataccel = np.sin(df["roll"].to_numpy(dtype=np.float32)) * 9.81
  v_ego = df["vEgo"].to_numpy(dtype=np.float32)
  a_ego = df["aEgo"].to_numpy(dtype=np.float32)
  target = df["targetLateralAcceleration"].to_numpy(dtype=np.float32)
  steer = -df["steerCommand"].to_numpy(dtype=np.float32)

  features = []
  labels = []
  valid_idx = np.flatnonzero(np.isfinite(steer))
  if len(valid_idx) == 0:
    return np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty((0,), dtype=np.float32)

  for idx in valid_idx:
    prev_target = target[idx - 1] if idx > 0 else target[idx]
    prev_action = steer[idx - 1] if idx > 0 else 0.0
    prev_prev_action = steer[idx - 2] if idx > 1 else prev_action
    features.append(
      build_policy_features(
        target_lataccel=target[idx],
        prev_target_lataccel=prev_target,
        future_lataccel=target[idx + 1:],
        roll_lataccel=roll_lataccel[idx],
        future_roll_lataccel=roll_lataccel[idx + 1:],
        v_ego=v_ego[idx],
        a_ego=a_ego[idx],
        prev_action=prev_action,
        prev_prev_action=prev_prev_action,
      )
    )
    labels.append(steer[idx])

  return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def build_dataset(files):
  x_parts = []
  y_parts = []
  for file_path in files:
    x_chunk, y_chunk = extract_examples(file_path)
    if len(x_chunk) == 0:
      continue
    x_parts.append(x_chunk)
    y_parts.append(y_chunk)
  if not x_parts:
    raise RuntimeError("No training examples extracted.")
  return np.vstack(x_parts), np.concatenate(y_parts)


def make_loader(x, y, batch_size, shuffle):
  dataset = TensorDataset(
    torch.from_numpy(x.astype(np.float32)),
    torch.from_numpy(y.astype(np.float32)),
  )
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model, loader, device):
  model.eval()
  total_loss = 0.0
  total_count = 0
  loss_fn = nn.SmoothL1Loss(beta=0.1, reduction="sum")
  with torch.no_grad():
    for features, targets in loader:
      features = features.to(device)
      targets = targets.to(device)
      preds = model(features)
      loss = loss_fn(preds, targets)
      total_loss += float(loss.item())
      total_count += int(targets.numel())
  return total_loss / max(total_count, 1)


def export_checkpoint(model, input_mean, input_std, out_path):
  state_dict = model.state_dict()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out_path,
    W1=state_dict["net.0.weight"].cpu().numpy().T,
    b1=state_dict["net.0.bias"].cpu().numpy(),
    W2=state_dict["net.2.weight"].cpu().numpy().T,
    b2=state_dict["net.2.bias"].cpu().numpy(),
    W3=state_dict["net.4.weight"].cpu().numpy().T,
    b3=state_dict["net.4.bias"].cpu().numpy(),
    input_mean=input_mean.astype(np.float32),
    input_std=input_std.astype(np.float32),
  )


def main():
  parser = argparse.ArgumentParser(description="Train BC steering policy from logged expert actions.")
  parser.add_argument("--num-segs", type=int, default=2000)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--batch-size", type=int, default=8192)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

  files = sorted(p for p in DATA_DIR.iterdir() if p.is_file())[:args.num_segs]
  if len(files) < 10:
    raise RuntimeError(f"Not enough segment files in {DATA_DIR}")

  split_idx = max(int(len(files) * 0.9), len(files) - 100)
  split_idx = min(max(split_idx, 1), len(files) - 1)
  train_files = files[:split_idx]
  val_files = files[split_idx:]

  print(f"Loading train dataset from {len(train_files)} segments...")
  x_train, y_train = build_dataset(train_files)
  print(f"Loading validation dataset from {len(val_files)} segments...")
  x_val, y_val = build_dataset(val_files)

  input_mean = x_train.mean(axis=0)
  input_std = x_train.std(axis=0)
  x_train = (x_train - input_mean) / np.maximum(input_std, 1e-6)
  x_val = (x_val - input_mean) / np.maximum(input_std, 1e-6)

  train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
  val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

  device = torch.device("cpu")
  model = BCPolicyNet(x_train.shape[1]).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  loss_fn = nn.SmoothL1Loss(beta=0.1)

  best_val = float("inf")
  best_state = None
  for epoch in range(1, args.epochs + 1):
    model.train()
    running = 0.0
    seen = 0
    for features, targets in train_loader:
      features = features.to(device)
      targets = targets.to(device)
      optimizer.zero_grad(set_to_none=True)
      preds = model(features)
      loss = loss_fn(preds, targets)
      loss.backward()
      optimizer.step()
      running += float(loss.item()) * int(targets.numel())
      seen += int(targets.numel())

    train_loss = running / max(seen, 1)
    val_loss = evaluate(model, val_loader, device)
    print(
      f"epoch={epoch} train_smoothl1={train_loss:.6f} "
      f"val_smoothl1={val_loss:.6f}"
    )

    if val_loss < best_val:
      best_val = val_loss
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

  if best_state is None:
    raise RuntimeError("Training produced no checkpoint.")
  model.load_state_dict(best_state)
  export_checkpoint(model, input_mean, input_std, args.output)
  print(f"saved={args.output}")


if __name__ == "__main__":
  main()
