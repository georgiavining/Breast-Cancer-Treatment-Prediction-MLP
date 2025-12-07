from sklearn.decomposition import PCA
import numpy as np

def apply_PCA(X_train, X_test, mandatory_indices, variance_retained):
    pca = PCA(n_components= variance_retained)

    if mandatory_indices is not None and len(mandatory_indices) > 0:

      mandatory_train = X_train[:, mandatory_indices]
      mandatory_test  = X_test[:, mandatory_indices]

      all_indices = np.arange(X_train.shape[1])
      pca_indices = [i for i in all_indices if i not in mandatory_indices]

      X_train_pca_input = X_train[:, pca_indices]
      X_test_pca_input  = X_test[:, pca_indices]

      X_train_pca = pca.fit_transform(X_train_pca_input)
      X_test_pca  = pca.transform(X_test_pca_input)
      n_components = pca.n_components_

      X_train_final = np.hstack([mandatory_train, X_train_pca])
      X_test_final = np.hstack([mandatory_test, X_test_pca])

    else:
      X_train_final = pca.fit_transform(X_train)
      X_test_final = pca.transform(X_test)
      n_components = pca.n_components_


    return X_train_final, X_test_final, pca, n_components