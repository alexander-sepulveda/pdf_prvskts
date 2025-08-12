# the goal of this program is showing how Gaussian Mixture Models work for the case
# of modeling Perovskite Solar Cells.

def toy_plot_marginal_conditionals(x, gmm, y_pred, y_pred_map, cov_type, n_points=200):
    """ Plot the marginal conditional distributions of each output dimension.
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

        ax.plot(xx, yy)
        ax.set_title(fr'conditional probability $\mathcal{{P}}\,(J_{{sc}} \mid E_g={x.item()})$')
        ax.set_xlabel('$J_{{sc}}$')
        ax.set_ylabel('probability')
        plt.axvline(x=y_pred, linestyle='--', color='green', linewidth=1.5, label='MSE estimate')
        plt.axvline(x=y_pred_map, linestyle='--', color='red', linewidth=1.5, label='local MAP estimate')
        ax.legend()

    plt.show()


def plot_inputs_vs_output(X, y, feature_names, y_name):
    import matplotlib.pyplot as plt
    n_features = X.shape[1]
    for i in range(n_features):
        plt.figure(figsize=(5, 4))
        plt.scatter(X[:, i], y)
        #plt.scatter(X[:, i], y)
        plt.xlabel(feature_names[i] if feature_names else f'X[{i}]')
        plt.ylabel(y_name)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_joinPDF_2D(weights, means, covariances, Xd, Yd, x_value):
    import numpy as np
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    min_x = np.min(Xd) - 0.5*np.std(Xd)
    max_x = np.max(Xd) + 0.5*np.std(Xd)
    min_y = np.min(Yd) - 0.5*np.std(Yd)
    max_y = np.max(Yd) + 0.5*np.std(Yd)

    x = np.linspace(min_x, max_x, 200)
    y = np.linspace(min_y, max_y, 200)

    XX, Y = np.meshgrid(x, y)
    pos = np.dstack((XX, Y))  # Shape: (200, 200, 2)
    # Compute total PDF from the mixture
    Z = np.zeros(XX.shape)
    for w, mu, cov in zip(weights, means, covariances):
        rv = multivariate_normal(mean=mu, cov=cov)
        Z += w * rv.pdf(pos)

    plt.figure(figsize=(5, 4))
    plt.scatter(Xd, Yd)
    contour = plt.contour(XX, Y, Z, levels=[0.01, 0.04, 0.08, 0.12, 0.15], cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    #plt.axvline(x=x, linestyle='--', color='black', linewidth=1.5, label= fr'$f \,\, (J_{{sc}} \, , \, E_g={float(x):.2f})$')
    plt.axvline(x=x_value, linestyle='--', color='black', linewidth=1.5, label=fr'$\mathcal{{P}} \,\, (J_{{sc}} \, , \, E_g={float(x_value[0])})$')
    #plt.axvline(x=y_pred, linestyle='--', color='green', linewidth=1.5, label='MSE estimation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.show()


def plot_joinPDF_3D(weights, means, covariances, Xd, Yd):
    import numpy as np
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    min_x = np.min(Xd) - 0.5*np.std(Xd)
    max_x = np.max(Xd) + 0.5*np.std(Xd)
    min_y = np.min(Yd) - 0.5*np.std(Yd)
    max_y = np.max(Yd) + 0.5*np.std(Yd)

    x = np.linspace(min_x, max_x, 200)
    y = np.linspace(min_y, max_y, 200)

    XX, Y = np.meshgrid(x, y)
    pos = np.dstack((XX, Y))  # Shape: (200, 200, 2)
    # Compute total PDF from the mixture
    Z = np.zeros(XX.shape)
    for w, mu, cov in zip(weights, means, covariances):
        rv = multivariate_normal(mean=mu, cov=cov)
        Z += w * rv.pdf(pos)

        Z = np.sqrt(Z)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel(x_names[0])
    ax.set_ylabel(y_name)
    ax.set_zlabel(fr"$\mathcal{{P}}\,(E_g, \, , \, J_{{sc}})$")
    plt.tight_layout()
    ax.set_zticklabels([])
    plt.show()

def toy_prediction(weights, means, covariances, Eg):
    # shows how to perform a simple calculation given the 2D model and the observed Band Gap.
    import numpy as np

    # === Parameters from equation (9) ===
    alpha1, alpha2 = weights[0], weights[1]
    mu1 = means[0]
    mu2 = means[1]
    C1  = covariances[0]
    C2  = covariances[1]

    mu1_x = mu1[0]
    mu2_x = mu2[0]
    C1_xx = C1[0, 0]
    C2_xx = C2[0, 0]

    # Gaussian PDFs for Eg
    pdf1_x = np.exp(-0.5 * (Eg - mu1_x)**2 / C1_xx) / np.sqrt(2 * np.pi * C1_xx)
    pdf2_x = np.exp(-0.5 * (Eg - mu2_x)**2 / C2_xx) / np.sqrt(2 * np.pi * C2_xx)

    # Responsibilities, equation (7)
    beta1 = (alpha1 * pdf1_x) / (alpha1 * pdf1_x + alpha2 * pdf2_x)
    beta2 = (alpha2 * pdf2_x) / (alpha1 * pdf1_x + alpha2 * pdf2_x)

    # Conditional means m1 and m2 (eq. 8)
    C1_yx = C1[1, 0]
    C2_yx = C2[1, 0]

    m1 = mu1[1] + C1_yx / C1_xx * (Eg - mu1_x)
    m2 = mu2[1] + C2_yx / C2_xx * (Eg - mu2_x)
    # Prediction y_hat (eq. 6)
    y_hat = beta1 * m1 + beta2 * m2

    # === Print results ===
    print("Eg [eV]:", Eg.item())
    print("N1(Eg): ", pdf1_x.item())
    print("N2(Eg): ", pdf2_x.item())
    print(f"β1 = {beta1.item():.2f}, β2 = {beta2.item():.2f}")
    print(f"m1 = {m1.item():.2f}, m2 = {m2.item():.2f}")
    print("Predicted Jsc [mA/cm²] :", np.round(y_hat.item(), 1))




if __name__ == "__main__":

    import pandas as PD
    import numpy as np
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    from pathlib import Path
    import sys
    sys.path.insert(0, './src')  # add src to Python path
    import GMM_utils as GMM
    import ABX

    Data = ABX.read_prvskt_data()

    embedding_dim = 4  # The desired embedding dimension

    var_input = ['Perovskite_band_gap']
    var_output = ['JV_default_Jsc']
    my_inputs = []
    my_inputs.append(var_input[0])
    my_inputs.append(var_output[0])
    my_2D_data = Data[my_inputs]
    my_2D_data = my_2D_data.drop_duplicates()

    #-- repeated values by dimension creates problems when using Log-Likelihood estimation criterion;
    #   thus, we eliminate a good part of them in case they are so many.
    keep_fraction = 0.05
    min_repeats = 20
    def reduce_group(group):
    	if len(group) > min_repeats:
    		k = max(1, int(len(group) * keep_fraction))
    		return group.sample(n=k, random_state=42)
    	else:
    		return group  # keep all values if <= 20

    df_reduced = my_2D_data.groupby('Perovskite_band_gap', group_keys=False).apply(reduce_group)
    my_2D_data = df_reduced

    # --- scatter plot ------------
    Inputs = my_2D_data[var_input]
    Outputs = my_2D_data[var_output]
    Xd = Inputs.to_numpy()
    Yd = Outputs.to_numpy()
    x_names = ['$E_g$ (bandgap)']
    y_name = '$J_{sc}$'
    plot_inputs_vs_output(Xd, Yd, x_names, y_name)   # plot s scatter plot.

    # ----------------------------------------------------------
    # Initialize and train the Gaussian Mixture Regression model
    n_inputs = len(var_input)
    n_outputs = len(var_output)
    N_mix = 2
    covariance_type = 'full'   # You can choose 'full', 'diagonal', 'spherical', or 'tied'
    data = np.hstack((Xd, Yd))
    gmr = GMM.GaussianMixtureRegression(N_mix, covariance_type)
    gmr.fit(data, n_inputs, n_outputs)   # Fit model
    gmm = gmr.gmm
    #-- the estimated parameters of the Prob. density function.
    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    print('\n')
    print('====  the model ====')
    print('* first normal component:')
    print(f'weight = {float(weights[0]):.2f}')
    print(f'mean = {np.round(means[0], 2)}')
    print(f'cov = {np.round(covariances[0], 3)}')
    print('\n')
    print('* second normal component:')
    print(f'weight = {float(weights[1]):.2f}')
    print(f'mean = {np.round(means[1], 2)}')
    print(f'cov = {np.round(covariances[1], 3)}')
    print('====================')
    print('\n')

    # assume a value for Band Gap before calculating the conditionals.
    Eg = np.array([1.4])
    plot_joinPDF_2D(weights, means, covariances, Xd, Yd, Eg)

    plot_joinPDF_3D(weights, means, covariances, Xd, Yd)

    y_pred, y_std, y_pred_map = gmr.predict(Eg)
    print('MSE estimate: ', y_pred)
    print('local MAP estimate: ',y_pred_map)
    print('\n')

    toy_plot_marginal_conditionals(Eg, gmm, y_pred, y_pred_map, cov_type='full', n_points=200)

    toy_prediction(weights, means, covariances, Eg)