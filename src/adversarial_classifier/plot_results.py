"""Plot training results from CSV history files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_model_results(csv_path: str, output_dir: str | None = None) -> None:
    """Plot train/val/test accuracy and loss for a single model."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    # Parse CSV
    epochs = []
    train_acc = []
    val_acc = []
    test_acc = []
    train_loss = []
    val_loss = []
    test_loss = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_acc.append(float(row['train_acc']) * 100)
            val_acc.append(float(row['val_acc']) * 100)
            test_acc.append(float(row['test_acc']) * 100)
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            test_loss.append(float(row['test_loss']))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(epochs, train_acc, marker='o', label='Train', linewidth=2)
    ax1.plot(epochs, val_acc, marker='s', label='Val', linewidth=2)
    ax1.plot(epochs, test_acc, marker='^', label='Test', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Model: {csv_path.stem.replace("_history", "")}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Plot loss
    ax2.plot(epochs, train_loss, marker='o', label='Train', linewidth=2)
    ax2.plot(epochs, val_loss, marker='s', label='Val', linewidth=2)
    ax2.plot(epochs, test_loss, marker='^', label='Test', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    out_dir = Path(output_dir or csv_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{csv_path.stem.replace('_history', '')}_chart.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Chart saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training history from CSV files")
    parser.add_argument("csv_files", nargs='+', help="CSV history files to plot")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for charts")
    args = parser.parse_args()

    for csv_file in args.csv_files:
        plot_model_results(csv_file, args.output_dir)


if __name__ == "__main__":
    main()
