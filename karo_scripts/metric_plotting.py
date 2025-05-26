import os
import json
import matplotlib.pyplot as plt
from collections import OrderedDict

def read_results(exp_dir):
    sequences = os.listdir(exp_dir)

    metrics = ["SSIM", "PSNR", "LPIPS-vgg", "LPIPS-alex", "MS-SSIM", "D-SSIM"]
    data_20000 = {metric: [] for metric in metrics}
    data_30000 = {metric: [] for metric in metrics}

    for seq in sequences:
        file_name = os.path.join(exp_dir, seq, 'results.json')

        # if file exists
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                json_data = json.load(f)
                for metric in metrics:
                    data_20000[metric].append(json_data["ours_20000"][metric])
                    data_30000[metric].append(json_data["ours_30000"][metric])
    return data_20000, data_30000, metrics

def iter_scatter(data_20000, data_30000, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.scatter(range(len(data_20000[metric])), data_20000[metric], color='blue', label='20000')
        ax.scatter(range(len(data_30000[metric])), data_30000[metric], color='red', label='30000')
        ax.set_title(metric)
        ax.set_xlabel("File index")
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()
    plt.show()

def read_per_frame(exp_dir):

    sequences = os.listdir(exp_dir)
    metrics = ["SSIM", "PSNR", "LPIPS-vgg", "LPIPS-alex", "MS-SSIM", "D-SSIM"]

    sorted_data = {}
    for seq in sequences:
        file_name = os.path.join(exp_dir, seq, 'per_view.json')

        # if file exists
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                json_data = json.load(f)

                sorted_data[seq] = {}
                for method, metrics in json_data.items():
                    sorted_data[seq][method] = {}
                    for metric, frame_dict in metrics.items():
                        sorted_data[seq][method][metric] = [pair[1] for pair in sorted(frame_dict.items())]
    return sorted_data

def plot_per_frame(sorted_data):

    metrics = ["SSIM", "PSNR", "LPIPS-vgg", "LPIPS-alex", "MS-SSIM", "D-SSIM"]
    labels = ['View 1', 'View 2', 'View 3', 'View 4']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    metric_directions = {
    "SSIM": "↑",
    "PSNR": "↑",
    "LPIPS-vgg": "↓",
    "LPIPS-alex": "↓",
    "MS-SSIM": "↑",
    "D-SSIM": "↓"}

    for seq_name, seq_data in sorted_data.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Metrics for sequence: {seq_name}", fontsize=16)
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            for method, method_data in seq_data.items():
                values = seq_data[method][metric]
                split = [values[j::4] for j in range(4)]  # Deinterleave

                for v in range(4):
                    ax.plot(split[v], label=f"{method} {labels[v]}", color=colors[v], alpha=0.6 if method == 'ours_20000' else 1.0, linestyle='-' if method == 'ours_30000' else '--')

            direction = metric_directions.get(metric, "")
            ax.set_title(f"{metric} ({direction})")
            ax.set_xlabel("Frame")
            ax.set_ylabel(metric)
            ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


if __name__ == "__main__":
    exp_dir = '/workspace/4DGaussians/output/mine_04/'

    # data_20000, data_30000, metrics = read_results(exp_dir)
    # iter_scatter(data_20000, data_30000, metrics)

    per_frame = read_per_frame(exp_dir)
    plot_per_frame(per_frame)