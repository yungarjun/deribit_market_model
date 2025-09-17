import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from typing import Tuple
from sklearn.cross_decomposition import PLSRegression

def pca_dyn_block(Z_train, n_dyn, seed):
    pca = PCA(n_components=n_dyn, random_state=seed).fit(Z_train)
    return pca, pca.components_

def pca_stat_block(R_resid, n_stat, seed):
    pca = PCA(n_components=n_stat, random_state=seed).fit(R_resid)
    return pca, pca.components_

def pda_from_pca(Z, pca_dyn):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    total_var = Zc.var(axis=0, ddof=1).sum()
    explained = pca_dyn.explained_variance_.sum()
    return 1.0 - (explained / total_var)

# try:
#     from sklearn.cross_decomposition import PLSRegression
# except Exception:
#     PLSRegression = None




def pls_block(X, Y, max_comp=10, n_splits=5, random_state=0):
    """
    Simple PLS with CV to choose #components. Returns (model, x_weights_ @ y_loadings_.T)
    so that the returned `factors` plays the role of loadings akin to PCA components.


    X: (T, N_features), Y: (T, N_targets). We will use Y=C (price residuals) and
    X=Z or previous residuals depending on your choice.
    """
    assert PLSRegression is not None, "scikit-learn's PLSRegression missing. pip install scikit-learn>=1.0"
    T = X.shape[0]
    best_k, best_score = 1, -np.inf
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for k in range(1, min(max_comp, min(X.shape[1], Y.shape[1], T-1)) + 1):
        scores = []
        for tr, te in kfold.split(X):
            pls = PLSRegression(n_components=k)
            pls.fit(X[tr], Y[tr])
            Yhat = pls.predict(X[te])
            # negative MSE as score
            mse = ((Y[te] - Yhat)**2).mean()
            scores.append(-mse)
        sc = float(np.mean(scores))
        if sc > best_score:
            best_score, best_k = sc, k


    pls = PLSRegression(n_components=best_k)
    pls.fit(X, Y)
# Derive factor-like matrix for compatibility: map scores to reconstruction
# Yhat = T_x * Q^T where T_x are X-scores; equiv factors for Y-space:
# Here we return Y_loadings_.T as a set of components acting on columns of Y
    factors = pls.y_loadings_.T # shape (k, n_targets)
    return pls, factors, best_k


class DeepPLS:
    """Minimal neural PLS surrogate: projects X->k scores with an encoder NN
    and reconstructs Y via linear loadings. Train with MSE + small ridge on
    loadings. This is a placeholder for experimentation, not production.
    """
    def __init__(self, n_in, n_out, k, hidden=128, lr=1e-3, epochs=200, seed=0, device=None):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, k)
        ).to(self.device)
        # loadings for Y reconstruction (k x n_out)
        self.loadings = nn.Parameter(torch.zeros(k, n_out, device=self.device))
        nn.init.xavier_uniform_(self.loadings)
        self.opt = torch.optim.Adam(list(self.encoder.parameters()) + [self.loadings], lr=lr)
        self.epochs = epochs


    def fit(self, X, Y):
        torch = self.torch
        X = torch.from_numpy(X).float().to(self.device)
        Y = torch.from_numpy(Y).float().to(self.device)
        for _ in range(self.epochs):
            T_x = self.encoder(X) # (T, k)
            Yhat = T_x @ self.loadings # (T, n_out)
            mse = ((Y - Yhat)**2).mean()
            ridge = 1e-4 * (self.loadings**2).mean()
            loss = mse + ridge
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return self


    def predict(self, X):
        torch = self.torch
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            T_x = self.encoder(X)
            Yhat = T_x @ self.loadings
        return Yhat.cpu().numpy()


    @property
    def factors(self):
    # return loadings like (k, n_out)
        return self.loadings.detach().cpu().numpy()