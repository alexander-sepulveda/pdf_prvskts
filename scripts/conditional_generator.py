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


def plot_2d_pdf_contour(X_gen_A, X_gen_B, var_idx1, var_idx2, grid_size=100):
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
    # --- Figure 1 ---
    # Create the FIRST figure and its axes
    fig1, ax1 = plt.subplots(figsize=(6, 5)) # (6,5) is a reasonable size for one
    c1 = ax1.contourf(x, y, z_A, levels=20, cmap='viridis')
    fig1.colorbar(c1, ax=ax1, label='probability')
    ax1.set_xlabel(r'$\chi_{sol.}$ (solvents ratio)')
    ax1.set_ylabel(r'$TB$ (thermal budget)')
    ax1.set_title(r'${\mathcal{P}}\,(\chi_{sol.}, TB \, \mid \, MAPbI_3, PCE=14)$')
    fig1.tight_layout() # Apply layout to the first figure
    # --- Figure 2 ---
    # Create the SECOND figure and its axes
    fig2, ax2 = plt.subplots(figsize=(6, 5)) # A separate figure
    c2 = ax2.contourf(x, y, z_B, levels=20, cmap='viridis')
    fig2.colorbar(c2, ax=ax2, label='probability')
    ax2.set_xlabel(r'$\chi_{sol.}$ (solvents ratio)')
    ax2.set_ylabel(r'$TB$ (thermal budget)')
    ax2.set_title(r'${\mathcal{P}}\,( \chi_{sol.}, TB \, \mid \, MAPbI_3, PCE=18)$')
    fig2.tight_layout() # Apply layout to the second figure
    # Show all created figures (this will open two separate windows)
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

    # ----------------------------------------------------------------------------
    # Load the saved model and scalers
    model_package = joblib.load('gmr_model_5_clusters.joblib')
    gmr_model = model_package['model']
    scaler_X  = model_package['scaler_x']
    scaler_y  =  model_package['scaler_y']

    # --- To obtain generated synthesis desriptors assuming we have the perovskite material and a desired PCE.
    observed_dims = [0, 1, 2, 3, 9]
    target_dims   = [4, 5, 6, 7, 8]
    z_material = [-0.036007170669086666, 0.006586621085668454, 0.01751554519205328, -0.0013882326952161786]   # Representation of MAPbI3

    z_pce = 14
    Zo = array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y)   #---- conditional input vector.
    cond_gmm = condition_gmm(gmr_model, observed_dims, target_dims, Zo)            # conditional gmm.
    new_samples = sample_from_conditional_gmm(cond_gmm, n_samples=2000, random_state=None)
    observed_dims = [0, 1, 2, 3]       # input observed dims.
    X_gen_A = inverse_transform_batch(new_samples, scaler_X, target_dims, observed_dims)

    observed_dims = [0, 1, 2, 3, 9]
    target_dims   = [4, 5, 6, 7, 8]
    z_material = [-0.036007170669086666, 0.006586621085668454, 0.01751554519205328, -0.0013882326952161786]   # Representation of MAPbI3
    z_pce = 18
    Zo = array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y)   #---- conditional input vector.
    cond_gmm = condition_gmm(gmr_model, observed_dims, target_dims, Zo)            # conditional gmm.
    new_samples = sample_from_conditional_gmm(cond_gmm, n_samples=2000, random_state=None)
    observed_dims = [0, 1, 2, 3]       # input observed dims.
    X_gen_B = inverse_transform_batch(new_samples, scaler_X, target_dims, observed_dims)
    #label_names = ['DMF DMSO ratio', 'annealing thermal budget', 'band gap', 'first annealing temperature', 'area_measured']
    var_idx1 = 0
    var_idx2 = 1
    plot_2d_pdf_contour(X_gen_A, X_gen_B, var_idx1, var_idx2)