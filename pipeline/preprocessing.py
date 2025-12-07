from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np


def preprocess_data(df, outcomes, target, test_size=0.2, random_state=42):
    """
    Preprocess the data by normalizing features and encoding labels.
    """

    #assigning features to X
    X = df.drop(outcomes, axis=1)
    #assigning target to y
    y = df[target]

    #splitting into test and train
    X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #imputing missing values
    imputer = IterativeImputer(random_state=random_state)
    X_train2 = imputer.fit_transform(X_train1)
    X_test2 = imputer.transform(X_test1)

    #scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train2)
    X_test = scaler.transform(X_test2)

    return X_train, X_test, y_train, y_test, imputer, scaler