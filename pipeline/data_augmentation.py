import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors

def augment_data_simple(X, y, n_aug=1, noise_std=0.01, random_state=42):
    """
    Augments data by adding Gaussian noise to existing samples.
    """

    np.random.seed(random_state)
    
    X_list = [X]
    y_list = [y]
    
    for _ in range(n_aug):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise
        X_list.append(X_noisy)
        y_list.append(y)
    
    X_aug = pd.concat([pd.DataFrame(x) for x in X_list], ignore_index=True)
    y_aug = pd.concat([pd.Series(label) for label in y_list], ignore_index=True)
    
    return X_aug, y_aug

def augment_data_smoter(X_train, y_train, rare_frac=0.2, k_neighbors=5):

    X_df = pd.DataFrame(X_train)
    y_series = pd.Series(y_train, name="target")
    df = pd.concat([X_df, y_series], axis=1)

    # Identify rare samples (top and bottom tails)
    lower_thresh = df['target'].quantile(rare_frac / 2)
    upper_thresh = df['target'].quantile(1 - rare_frac / 2)
    rare_df = df[(df['target'] <= lower_thresh) | (df['target'] >= upper_thresh)]

    X_rare = rare_df.drop(columns=['target']).values
    y_rare = rare_df['target'].values

    if len(X_rare) == 0:
        print("No rare samples found for augmentation.")
        return X_train, y_train

    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X_rare))).fit(X_rare)

    # Generate synthetic samples
    X_synth = []
    y_synth = []
    for i in range(len(X_rare)):
        distances, indices = nn.kneighbors([X_rare[i]])
        for idx in indices[0]:
            if idx != i:  # skip self
                lam = np.random.rand()
                X_new = lam * X_rare[i] + (1 - lam) * X_rare[idx]
                y_new = lam * y_rare[i] + (1 - lam) * y_rare[idx]
                X_synth.append(X_new)
                y_synth.append(y_new)

    # Combine original + synthetic
    X_aug = np.vstack([X_train, np.array(X_synth)])
    y_aug = np.hstack([y_train, np.array(y_synth)])

    # Shuffle
    X_aug, y_aug = shuffle(X_aug, y_aug)
    print(f"Original train size: {X_train.shape[0]}")
    print(f"Augmented train size: {X_aug.shape[0]}")

    return X_aug, y_aug

      