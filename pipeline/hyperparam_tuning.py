import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import os
import time

#setting up logs directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    run_dir = os.path.join(LOGS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

run_logdir = get_run_logdir()

def run_hyperband_tuner(build_fn, input_dim, X_train, y_train, project_name,
                        max_epochs=30, factor=3, seed=42, validation_split=0.2,
                        verbose=1):
    
    # initialising tuner
    tuner = kt.Hyperband(
        lambda hp: build_fn(hp, input_dim),
        objective='val_mae',
        max_epochs=max_epochs,
        factor=factor,
        directory='keras_tuner_logs',
        project_name=project_name,
        overwrite=False,
        seed=seed
    )

    #callbacks
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=os.path.join(run_logdir, project_name)
)
    callbacks = [earlystop_cb, tensorboard_cb]

    #running search
    tuner.search(
        X_train, y_train,
        epochs=max_epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )

    return tuner
