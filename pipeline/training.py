from tensorflow import keras
import os
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

#directories setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
RESULTS_DIR = os.path.join(BASE_DIR, "results")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_dir = os.path.join(RESULTS_DIR, f"experiment_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

plots_dir = os.path.join(experiment_dir, "training_plots")
os.makedirs(plots_dir, exist_ok=True)


def train_and_save(name, model, hp, X_train, y_train, X_test, y_test, epochs=80):
    """
    Trains a PCA model, evaluates test metrics, and saves everything.
    """

    pca_dir = os.path.join(experiment_dir, name)
    os.makedirs(pca_dir, exist_ok=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.1,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    test_loss, test_mae = model.evaluate(X_test, y_test)

    metrics = {
        "train_mae": history.history['loss'][-1],
        "val_mae": history.history['val_loss'][-1],
        "test_mae": test_mae
    }
    
    model_path = os.path.join(pca_dir, "best_model.keras")
    model.save(model_path)

    hp_path = os.path.join(pca_dir, "hyperparams.json")
    with open(hp_path, "w") as f:
        json.dump(hp.values, f, indent=4)

    metrics_json_path = os.path.join(pca_dir, "metrics.json")
    metrics_txt_path = os.path.join(pca_dir, "metrics.txt")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(metrics_txt_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"[{name}] Saved model and metrics -> {pca_dir}")
    return history, metrics

def plot_history(name, history, title):
    """
    Plots training and validation loss over epochs.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plot_path = os.path.join(plots_dir, f"{name}_training_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')

    plt.show()
    plt.close()