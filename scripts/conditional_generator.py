def condition_gmm(gmm, observed_dims, target_dims, z_observed):
    """  Compute the conditional GMM: p(Z_target | Z_observed = z_observed)
    Parameters:
        gmm: trained sklearn GaussianMixture model (on full D-dim space)
        observed_dims: list of indices for observed variables (Z_a)
        target_dims: list of indices for target variables (Z_b)
        z_observed: array-like, shape (len(observed_dims),)
    Returns:
        conditional_gmm: dict with keys 'weights', 'means', 'covariances' representing the conditional GMM over Z_b
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from scipy.stats import multivariate_normal

    K = gmm.N_mix
    weights = []
    cond_means = []
    cond_covs = []

    for k in range(K):
        mu = gmm.gmm.means_[k]
        cov = gmm.gmm.covariances_[k]

        # Partition the mean and covariance
        mu_a = mu[observed_dims]
        mu_b = mu[target_dims]

        Sigma_aa = cov[np.ix_(observed_dims, observed_dims)]
        Sigma_bb = cov[np.ix_(target_dims, target_dims)]
        Sigma_ab = cov[np.ix_(target_dims, observed_dims)]
        Sigma_ba = Sigma_ab.T

        # Invert Sigma_aa safely
        Sigma_aa_inv = np.linalg.pinv(Sigma_aa)

        # Conditional mean and covariance
        cond_mu = mu_b + Sigma_ab @ Sigma_aa_inv @ (z_observed - mu_a)
        cond_cov = Sigma_bb - Sigma_ab @ Sigma_aa_inv @ Sigma_ba

        cond_means.append(cond_mu)
        cond_covs.append(cond_cov)

        # Compute conditional weight (unnormalized)
        w_k = gmm.gmm.weights_[k]
        p_z_a = multivariate_normal(mean=mu_a, cov=Sigma_aa).pdf(z_observed)
        weights.append(w_k * p_z_a)

    # Normalize weights
    weights = np.array(weights)
    weights /= np.sum(weights)

    return {
        'weights': weights,
        'means': cond_means,
        'covariances': cond_covs
    }



def sample_from_conditional_gmm(cond_gmm, n_samples=1, random_state=None):
    import numpy as np

    rng = np.random.default_rng(random_state)
    weights = cond_gmm['weights']
    means = cond_gmm['means']
    covariances = cond_gmm['covariances']
    n_components = len(weights)
    dim = len(means[0])

    # Choose which component each sample comes from
    component_indices = rng.choice(n_components, size=n_samples, p=weights)

    samples = np.zeros((n_samples, dim))
    for i, comp in enumerate(component_indices):
        samples[i] = rng.multivariate_normal(means[comp], covariances[comp])

    return samples

def array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y):
    z_material = np.array(z_material)
    zeros = np.zeros(len(target_dims))
    z_material = np.append(z_material, zeros)
    target_dims = np.array(target_dims)
    if z_material.ndim == 1:
        z_o = z_material.reshape(1, -1)  # (1, 9)
        z_o = scaler_X.transform(z_o)    # standardize
        Zo = np.delete(z_o, target_dims, axis=1)  # shape (1, 4)
        y_o = np.array([[z_pce]])               # shape (1, 1)
        y_o = scaler_y.transform(y_o)           # shape (1, 1)
        Zo = np.hstack((Zo, y_o))               # final shape (1, 5)
        Zo = Zo.flatten()

    return Zo


def inverse_transform_batch(X_target, scaler, target_dims, observed_dims):
    """ Inverse-transform a batch of p-dimensional vectors back to original scale
    using a StandardScaler trained on full N-dimensional data.
    Parameters:
        X_target : Standardized vectors containing only values for target_dims.
        scaler : fitted StandardScaler. Fitted on N-dimensional input.
        target_dims : list of int. Indices of the known dimensions.
        observed_dims : list of int. Indices of the zero-filled dimensions (used for padding).
    Returns:
        X_target_inv : Inverse-transformed (original scale) values for the target_dims.
    """
    n_samples, p = X_target.shape
    N = len(target_dims) + len(observed_dims)

    # Create zero-padded full matrix
    X_full = np.zeros((n_samples, N))
    X_full[:, target_dims] = X_target

    X_full_inv = scaler.inverse_transform(X_full)
    X_target_inv = X_full_inv[:, target_dims]

    return X_target_inv


def plot_2d_pdf_contour(X_gen_A, X_gen_B, label_names, var_idx1, var_idx2, grid_size=100):
    """ Plot 2D PDF contour for two selected variables from multivariate data.
    Parameters:
        data      : np.ndarray of shape (n_samples, p)
        var_idx1  : int, index of first variable
        var_idx2  : int, index of second variable
        grid_size : resolution of contour grid
    """
    # 1. Extract 2D projection
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    data_2d = X_gen_A[:, [var_idx1, var_idx2]].T  # shape (2, n_samples)

    x_min, x_max = data_2d[0].min(), data_2d[0].max()
    y_min, y_max = data_2d[1].min(), data_2d[1].max()
    x, y = np.mgrid[x_min:x_max:grid_size*1j, y_min:y_max:grid_size*1j]
    grid_coords = np.vstack([x.ravel(), y.ravel()])

    kde_A = gaussian_kde(data_2d)
    z_A = kde_A(grid_coords).reshape(grid_size, grid_size)

    data_B = X_gen_B[:, [var_idx1, var_idx2]].T  # shape (2, n_samples)
    kde_B = gaussian_kde(data_B)
    z_B = kde_B(grid_coords).reshape(grid_size, grid_size)

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # two "figures" side-by-side
    # First "figure"
    c1 = axes[0].contourf(x, y, z_A, levels=20, cmap='viridis')
    fig.colorbar(c1, ax=axes[0], label='Density')
    axes[0].set_xlabel('DMF DMSO ratio')
    axes[0].set_ylabel('annealing thermal budget')
    axes[0].set_title('Sample A')
    # Second "figure"
    c2 = axes[1].contourf(x, y, z_B, levels=20, cmap='viridis')
    fig.colorbar(c2, ax=axes[1], label='Density')
    axes[1].set_xlabel('DMF DMSO ratio')
    axes[1].set_ylabel('annealing thermal budget')
    axes[1].set_title('Sample B')
    plt.tight_layout()
    plt.show()



def compare_tsne_kde_from_arrays(real_embedded, generated_embedded, bw_adjust, label_names=('Real', 'Generated'), colors=('blue', 'orange')):
    """ Plot 1D KDEs for each t-SNE component, comparing real vs generated data.
    Parameters:
    real_embedded : t-SNE projection of real data.
    generated_embedded : t-SNE projection of generated data.
    label_names : Labels for the real and generated data.
    colors : Colors to use in the plots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for dim in [0, 1]:
        ax = axes[dim]
        sns.kdeplot(real_embedded[:, dim], fill=True, label=label_names[0],
                    color=colors[0], linewidth=1, ax=ax, bw_adjust=bw_adjust)
        sns.kdeplot(generated_embedded[:, dim], fill=True, label=label_names[1],
                    color=colors[1], linewidth=1, ax=ax, bw_adjust=bw_adjust)

        ax.set_title(f't-SNE Dimension {dim + 1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    import pandas as PD
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.model_selection import KFold

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    import joblib  # for saving and loading models

    from pathlib import Path
    import sys
    sys.path.insert(0, './src')  # add src to Python path
    import GMM_utils as GMM
    import ABX

    base_dir = Path(__file__).parent.parent    # project location

    Data, Inputs, Outputs = ABX.read_prvskt_data()
    X = Inputs.to_numpy()
    Y = Outputs.to_numpy()
    # ----------------------------------
    embedding_dim = 4  # The desired embedding dimension

    observed_dims = [0, 1, 2, 3, 9]   # assume we select the perovskite material and PCE;
    target_dims   = [4, 5, 6, 7, 8]   # and, we want to generate systhesis conditions.
    
    # load model and the indexes.
    closest_index = 9
    gmr = joblib.load(base_dir / "temp" / f"gmm_cv_fold{closest_index+1}.joblib")
    test_index = joblib.load(base_dir / "temp" / f"test_indexes{closest_index+1}.joblib")
    train_index = joblib.load(base_dir / "temp" / f"train_indexes{closest_index+1}.joblib")
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(Y_train)

    # ---------------------------------------------------------------
    # To probabilistically compare the generated data and the actual data.
    samples, labels = gmr.gmm.sample(550)
    if Y_test.ndim == 1:
        Y_test = Y_test.reshape(-1, 1)
    Z_real = np.hstack((X_test, Y_test))  # Make sure X_test and Y_test have compatible shapes too

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    Z_gen_emb = tsne.fit_transform(samples)  # shape: (n_samples, 2)
    Z_real_emb = tsne.fit_transform(Z_real)  # shape: (n_samples, 2)
    bw_adjust = 3
    compare_tsne_kde_from_arrays(Z_real_emb, Z_gen_emb, bw_adjust, label_names=('Real', 'Generated'), colors=('blue', 'orange'))

    from scipy.stats import ks_2samp

    for dim in range(Z_gen_emb.shape[1]):
        stat, pval = ks_2samp(Z_gen_emb[:, dim], Z_real_emb[:, dim])
        print(f"t-SNE Dimension {dim + 1}:")
        print(f"  KS statistic = {stat:.4f}")
        print(f"  p-value      = {pval:.4f}")
        if pval < 0.05:
            print("  ❌ Distributions differ significantly (reject H0)\n")
        else:
            print("  ✅ No significant difference (fail to reject H0)\n")

    # ----------------------------------------------------------------------------
    # --- To obtain generated synthesis desriptors assuming we have the perovskite material and a desired PCE.
    observed_dims = [0, 1, 2, 3, 9]
    target_dims   = [4, 5, 6, 7, 8]
    z_material = [-0.036007170669086666, 0.006586621085668454, 0.01751554519205328, -0.0013882326952161786]   # Representation of MAPbI3

    z_pce = 20
    Zo = array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y)   #---- conditional input vector.
    cond_gmm = condition_gmm(gmr, observed_dims, target_dims, Zo)            # conditional gmm.
    new_samples = sample_from_conditional_gmm(cond_gmm, n_samples=2000, random_state=None)
    observed_dims = [0, 1, 2, 3]       # input observed dims.
    X_gen_A = inverse_transform_batch(new_samples, scaler_X, target_dims, observed_dims)

    observed_dims = [0, 1, 2, 3, 9]
    target_dims   = [4, 5, 6, 7, 8]
    z_material = [-0.036007170669086666, 0.006586621085668454, 0.01751554519205328, -0.0013882326952161786]   # Representation of MAPbI3
    z_pce = 11
    Zo = array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y)   #---- conditional input vector.
    cond_gmm = condition_gmm(gmr, observed_dims, target_dims, Zo)            # conditional gmm.
    new_samples = sample_from_conditional_gmm(cond_gmm, n_samples=2000, random_state=None)
    observed_dims = [0, 1, 2, 3]       # input observed dims.
    X_gen_B = inverse_transform_batch(new_samples, scaler_X, target_dims, observed_dims)

    label_names = ['DMF DMSO ratio', 'annealing thermal budget', 'band gap', 'first annealing temperature', 'area_measured']
    var_idx1 = 0
    var_idx2 = 1
    plot_2d_pdf_contour(X_gen_A, X_gen_B, label_names, var_idx1, var_idx2)