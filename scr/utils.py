import subprocess
import torch
import os

def get_available_gpu():
    """
    Auto-detect the least busy GPU on the server and set it for torch.
    """
    try:
        # Query nvidia-smi for GPU utilization
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        # Parse output
        gpu_stats = []
        for line in result.strip().split("\n"):
            index, mem_used, mem_total = map(int, line.split(","))
            gpu_stats.append((index, mem_used, mem_total))

        # Sort by least memory used
        gpu_stats.sort(key=lambda x: x[1])  # sort by mem_used

        # Pick first GPU
        best_gpu = gpu_stats[0][0]

        # Set CUDA_VISIBLE_DEVICES to use only this GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

        # Now only one GPU will be visible to PyTorch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Selected GPU: {best_gpu} | Memory Used: {gpu_stats[0][1]} MB")
        return device

    except Exception as e:
        print(f"GPU detection failed: {e}")
        print("Defaulting to CPU.")
        return torch.device("cpu")


# Example usage
device = get_available_gpu()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to count as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
