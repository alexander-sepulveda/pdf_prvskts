class GaussianMixtureRegression:

    def __init__(self, n_components=5, covariance_type='full', random_state=None):
        self.N_mix = n_components
        self.cov_type = covariance_type
        self.rand_state = random_state

    def fit(self, data, input_dim, output_dim):
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_dim = input_dim + output_dim

        # Initialize GMM using KMeans
        kmeans = KMeans(n_clusters=self.N_mix, random_state=self.rand_state, n_init = 10)
        kmeans.fit(data)

        self.gmm = GaussianMixture(n_components=self.N_mix,
            covariance_type=self.cov_type,
            random_state=self.rand_state,
            means_init=kmeans.cluster_centers_,
            max_iter=50,
            n_init=10,
            init_params='kmeans'
            )
        self.gmm.fit(data)

    def predict(self, X):
        import numpy as np

        n_samples = X.shape[0]
        means = self.gmm.means_
        covariances = self.gmm.covariances_
        weights = self.gmm.weights_

        Y_pred_mse = np.zeros((n_samples, self.output_dim))
        Y_pred_map = np.zeros((n_samples, self.output_dim))
        Y_pred_std = np.zeros((n_samples, self.output_dim))  # Full covariance
        
        for i, x in enumerate(X):
            # MMSE estimation.
            y_hat, cond_var = compute_GMM_MMSE_estimate(x, self.gmm, self.cov_type)
            Y_pred_mse[i] = y_hat
            cond_var_diag = np.diag(cond_var)
            Y_pred_std[i] = np.sqrt(cond_var_diag)

            # MAP estimation.
            y_map = compute_gmm_map_estimate(x, self.gmm, self.cov_type)
            Y_pred_map[i] = y_map

        return Y_pred_mse, Y_pred_std, Y_pred_map


# ----- functions to calculate the MMSE Estimate.
def compute_GMM_MMSE_estimate(x, gmm, cov_type):
    import numpy as np
    from scipy.stats import multivariate_normal

    x = x.reshape(1, -1)
    input_dim = len(x[0])
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    N_mix = len(gmm.weights_)   # number of mixtures.
    total_dim = len(means[0])
    output_dim  = total_dim - input_dim
    
    cond_means = []
    cond_covs = []   #-
    cond_weights = np.array([])
    cond_samples = []
    for k in range(N_mix):
        mu = means[k]
        mu_x = mu[:input_dim]
        mu_y = mu[input_dim:]

        if cov_type == 'full':
            cov = covariances[k]
        elif cov_type == 'diag':
            cov = np.diag(covariances[k])
        elif cov_type == 'tied':
            cov = covariances
        elif cov_type == 'spherical':
            cov = np.eye(total_dim) * covariances[k]
        else:
            raise ValueError("Unsupported covariance_type")

        cov_xx = cov[:input_dim, :input_dim] + 1e-5*np.eye(input_dim)  # Regularization to avoid singular matrix
        cov_xy = cov[:input_dim, input_dim:]
        cov_yx = cov[input_dim:, :input_dim]
        cov_yy = cov[input_dim:, input_dim:]

        # Conditional mean
        try:
            inv_cov_xx = np.linalg.inv(cov_xx)
        except np.linalg.LinAlgError:
            inv_cov_xx = np.linalg.pinv(cov_xx)

        inv_cov_xx = np.linalg.inv(cov_xx)
        mu_y = mu_y.reshape(-1, 1)
        cond_mu = mu_y + cov_yx @ inv_cov_xx @ (x - mu_x).T
        cond_cov = cov_yy - cov_yx @ inv_cov_xx @ cov_xy         #-
        cond_means.append(cond_mu)
        cond_covs.append(cond_cov)    #-

        # Responsibilities
        rv = multivariate_normal(mean=mu_x, cov=cov_xx)
        resp = weights[k] * rv.pdf(x.flatten())
        cond_weights = np.append(cond_weights, resp)

    # MMSE estimation:
    sum_weights = np.sum(cond_weights)
    cond_weights = cond_weights / (sum_weights + 1e-13 )
    cond_means_matrix = np.hstack(cond_means).T
    y_hat = cond_weights @ cond_means_matrix

    # Compute conditional variance
    cond_var = np.zeros((output_dim, output_dim))
    for k in range(N_mix):
        mu_k = cond_means[k].flatten()
        diff = (mu_k - y_hat).reshape(-1, 1)
        cov_k = cond_covs[k]
        cond_var += cond_weights[k] * (cov_k + diff @ diff.T)
    
    return y_hat, cond_var


# ----- functions to calculate the MAP Estimate.
def compute_gmm_map_estimate(x, gmm, cov_type):
    """  Computes the MAP estimate of Y | x from a GMM model.
    Parameters:
    - x: np.array of shape (d_x,), the input vector
    - gmm_params: list of dicts with keys:
        'pi': float, mixture weight
        'mu': np.array (d_x + d_y,), full mean vector
        'cov': np.array ((d_x + d_y, d_x + d_y)), full covariance matrix
    Returns:
    - y_map: np.array of shape (d_y,), the MAP estimate of Y given x
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import multivariate_normal

    x = x.reshape(1, -1)

    input_dim = len(x[0])
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_
    N_mix = len(gmm.weights_)   # number of mixtures.
    total_dim = len(means[0])
    output_dim  = total_dim - input_dim

    pesos = []
    cond_means = []
    cond_covs = []

    for k in range(N_mix):
        pi_k = weights[k]
        mu   = means[k]
        cov  = covariances[k]
        mu_x = mu[:input_dim]
        mu_y = mu[input_dim:]

        if cov_type == 'full':
            cov = covariances[k]
        elif cov_type == 'diag':
            cov = np.diag(covariances[k])
        elif cov_type == 'tied':
            cov = covariances
        elif cov_type == 'spherical':
            cov = np.eye(total_dim) * covariances[k]
        else:
            raise ValueError("Unsupported covariance_type")

        cov_xx = cov[:input_dim, :input_dim] + 1e-4*np.eye(input_dim)  # Regularization to avoid singular matrix
        cov_xy = cov[:input_dim, input_dim:]
        cov_yx = cov[input_dim:, :input_dim]
        cov_yy = cov[input_dim:, input_dim:]

        try:
            inv_cov_xx = np.linalg.inv(cov_xx)
        except np.linalg.LinAlgError:
            inv_cov_xx = np.linalg.pinv(cov_xx)

        #inv_cov_xx = np.linalg.inv(cov_xx)
        x = x.reshape(-1)
        mu_x = mu_x.reshape(-1)
        diff = x - mu_x                # shape: (d_x,)
        cond_mu = mu_y + cov_yx @ inv_cov_xx @ diff  # shape: (d_y,)
        cond_cov = cov_yy - cov_yx @ inv_cov_xx @ cov_xy

        marginal = multivariate_normal(mean=mu_x, cov=cov_xx).pdf(x)
        pesos.append(pi_k * marginal)
        cond_means.append(cond_mu)
        cond_covs.append(cond_cov)

    pesos = np.array(weights)
    pesos /= np.sum(weights)

    # Negative log of the posterior mixture (to minimize)
    def negative_mixture_log_pdf(y):
        pdf_val = sum(
            w * multivariate_normal.pdf(y, mean=mu, cov=cov, allow_singular=True)
            for w, mu, cov in zip(weights, cond_means, cond_covs)
        )
        return -np.log(pdf_val + 1e-12)  # Add epsilon to avoid log(0)

    # Start optimization at the conditional mean of the most probable component
    best_k = np.argmax(weights)
    y0 = cond_means[best_k]

    res = minimize(negative_mixture_log_pdf, x0=y0, method='L-BFGS-B')

    if not res.success:
        print("Warning: Optimization did not converge.")

    y_map = res.x
    return y_map

#-----------------------------
def plot_marginal_conditionals(x, gmm, cov_type, n_points=200):
    """
    Plot the marginal conditional distributions of each output dimension.
    
    Parameters:
        x          : Input vector (1D or 2D of shape (1, d_x))
        gmm        : Fitted sklearn.mixture.GaussianMixture
        cov_type   : Covariance type ('full', 'diag', 'tied', 'spherical')
        n_points   : Number of points in x-axis for plotting each marginal
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    x = x.reshape(1, -1)
    input_dim = x.shape[1]
    total_dim = gmm.means_.shape[1]
    output_dim = total_dim - input_dim

    cond_means = []
    cond_covs = []
    cond_weights = []

    for k in range(gmm.n_components):
        mu = gmm.means_[k]
        mu_x = mu[:input_dim]
        mu_y = mu[input_dim:]

        # Covariance handling
        if cov_type == 'full':
            cov = gmm.covariances_[k]
        elif cov_type == 'diag':
            cov = np.diag(gmm.covariances_[k])
        elif cov_type == 'tied':
            cov = gmm.covariances_
        elif cov_type == 'spherical':
            cov = np.eye(total_dim) * gmm.covariances_[k]
        else:
            raise ValueError("Unsupported covariance_type")

        cov_xx = cov[:input_dim, :input_dim] + 1e-6 * np.eye(input_dim)
        cov_xy = cov[:input_dim, input_dim:]
        cov_yx = cov[input_dim:, :input_dim]
        cov_yy = cov[input_dim:, input_dim:]

        inv_cov_xx = np.linalg.pinv(cov_xx)
        diff = (x - mu_x).reshape(-1, 1)

        cond_mu = mu_y.reshape(-1, 1) + cov_yx @ inv_cov_xx @ diff
        cond_cov = cov_yy - cov_yx @ inv_cov_xx @ cov_xy

        cond_means.append(cond_mu.flatten())
        cond_covs.append(np.diag(cond_cov))  # take marginal variances
        weight = gmm.weights_[k] * norm.pdf(x.flatten(), loc=mu_x, scale=np.sqrt(np.diag(cov_xx))).prod()
        cond_weights.append(weight)

    # Normalize weights
    cond_weights = np.array(cond_weights)
    cond_weights /= cond_weights.sum()

    # Plot each output dimension
    fig, axes = plt.subplots(output_dim, 1, figsize=(6, 3 * output_dim), constrained_layout=True)
    if output_dim == 1:
        axes = [axes]

    for d in range(output_dim):
        ax = axes[d]

        # Compute global range
        all_means = [cond_means[k][d] for k in range(gmm.n_components)]
        all_stds = [np.sqrt(cond_covs[k][d]) for k in range(gmm.n_components)]
        min_x = min(all_means) - 3 * max(all_stds)
        max_x = max(all_means) + 3 * max(all_stds)

        xx = np.linspace(min_x, max_x, n_points)
        yy = np.zeros_like(xx)

        for k in range(gmm.n_components):
            mu_k = cond_means[k][d]
            std_k = np.sqrt(cond_covs[k][d])
            weight_k = cond_weights[k]
            yy += weight_k * norm.pdf(xx, mu_k, std_k)

        ax.plot(xx, yy, label=f'Marginal PDF of output {d+1}')
        ax.set_title(f'Marginal Conditional PDF - Output {d+1}')
        ax.set_xlabel('Output value')
        ax.set_ylabel('Density')
        ax.legend()

    plt.show()