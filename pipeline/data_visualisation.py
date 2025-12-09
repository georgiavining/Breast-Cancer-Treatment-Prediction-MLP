import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_distributions(df, features, target):
    """
    Plots histograms for features and the target.
    """
    import math
    n_features = len(features)
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)

    plt.figure(figsize=(4*n_cols, 3*n_rows))
    for i, col in enumerate(features):
        plt.subplot(n_rows, n_cols, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6,4))
    sns.histplot(df[target], kde=True, color='orange')
    plt.title(f"{target} distribution")
    plt.show()

def plot_target_distribution(y, title="RFS Distribution"):
    plt.figure(figsize=(8,5))
    sns.histplot(y, bins=30, kde=True, color="skyblue")
    plt.title(title)
    plt.xlabel("RFS Value")
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation(df, features, target):
    """
    Plots a correlation heatmap of features against the target.
    """
    corr = features.copy()
    corr[target] = df[target]  
    corr_matrix = corr.corr()
    
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix.drop(index=target).drop(columns=target), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    print("Correlation with target:")
    print(corr_matrix[target].sort_values(ascending=False))
