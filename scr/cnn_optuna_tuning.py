import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import time
import csv
import os

from baseline_analysis import split_training_data
from utils import get_available_gpu, EarlyStopping
import optuna.visualization as vis

device = get_available_gpu()

# --- Baseline CNN ---
class BaselineCNN(nn.Module):
    def __init__(self, n_filters=16, dropout=0.25, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        in_channels = 3
        for _ in range(num_layers):
            self.convs.append(nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1))
            in_channels = n_filters
            n_filters *= 2
        self.pool = nn.MaxPool2d(2)
        self.flatten_dim = in_channels * 56 * 56
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(torch.relu(conv(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.out(x)

# --- Objective Function ---
def objective(trial):
    # Expanded search space
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-1)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_filters = trial.suggest_categorical("n_filters", [16, 32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 6)

    image_dir = "/mnt/research-projects/j/jlgage/RawUAVData01/data/images"
    csv_path = "/mnt/research-projects/j/jlgage/RawUAVData01/data/all_scored_images_clean.csv"
    train_loader, val_loader, _ = split_training_data(image_dir, csv_path, batch_size=batch_size)

    model = BaselineCNN(n_filters, dropout, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    early_stopper = EarlyStopping(patience=3, min_delta=0.001)
    start_time = time.time()
    best_val_loss = float("inf")
    epoch_ran = 0
    early_stopped = False

    for epoch in range(20):
        epoch_ran += 1
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_trial_{trial.number}.pt")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            early_stopped = True
            break

    trial.set_user_attr("epochs", epoch_ran)
    trial.set_user_attr("early_stopped", early_stopped)
    trial.set_user_attr("duration_sec", round(time.time() - start_time, 2))

    return best_val_loss

# --- CSV Logger ---
def log_trial_to_csv(trial, filepath="cnn_trials_log.csv"):
    header = list(trial.params.keys()) + ["value", "epochs", "early_stopped", "duration_sec"]
    row = [trial.params.get(k, "NA") for k in trial.params] + [
        trial.value,
        trial.user_attrs.get("epochs", "NA"),
        trial.user_attrs.get("early_stopped", "NA"),
        trial.user_attrs.get("duration_sec", "NA")
    ]

    write_header = not os.path.exists(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# --- Callback ---
def logging_callback(study, trial):
    log_trial_to_csv(trial)

# --- Run Optimization and Plot ---
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, callbacks=[logging_callback])

    print("Best trial:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    print("Epochs:", study.best_trial.user_attrs["epochs"])
    print("Early Stopped:", study.best_trial.user_attrs["early_stopped"])
    print("Duration (sec):", study.best_trial.user_attrs["duration_sec"])

    # Save interactive visualizations
    vis.plot_optimization_history(study).write_html("cnn_opt_history.html")
    vis.plot_param_importances(study).write_html("cnn_param_importance.html")
    vis.plot_parallel_coordinate(study).write_html("cnn_parallel_coords.html")
    vis.plot_contour(study).write_html("cnn_contour_plot.html")
    vis.plot_slice(study).write_html("cnn_slice_plot.html")

