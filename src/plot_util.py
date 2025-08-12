
def plot_pdfs_inputs(Data):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    swedish_blue = '#005293'  # Hex color
    #'DMF_DMSO_ratio'
    sns.kdeplot(Data['DMF_DMSO_ratio'], fill=True, linewidth=2, color=swedish_blue)
    plt.xlabel(r'$\chi_{sol.}$')
    plt.ylabel('Probability')
    plt.show()

    #'first_Prvskt_annealing_temperature'
    sns.kdeplot(Data['first_Prvskt_annealing_temperature'], fill=True, linewidth=2, color=swedish_blue)
    plt.xlabel(r'$T_1$')
    plt.ylabel('Probability')
    plt.show()

    #'Perovskite_annealing_thermal_exposure'
    sns.kdeplot(Data['Perovskite_annealing_thermal_exposure'], fill=True, linewidth=2, color=swedish_blue)
    plt.xlabel(r'$TB$')
    plt.ylabel('Probability')
    plt.show()

    #'Perovskite_band_gap'
    sns.kdeplot(Data['Perovskite_band_gap'], fill=True, linewidth=2, color=swedish_blue)
    plt.xlabel(r'$E_g$')
    plt.ylabel('Probability')
    plt.show()


    #'Cell_area_measured'
    sns.kdeplot(Data['Cell_area_measured'], fill=True, linewidth=2, color=swedish_blue)
    plt.xlabel(r'$\mathcal{A}$')
    plt.ylabel('Probability')
    plt.show()

    #'JV_default_PCE'
    sns.kdeplot(Data['JV_default_PCE'], fill=True, linewidth=2, color=swedish_blue)
    plt.xlabel('PCE')
    plt.ylabel('Probability')
    plt.show()

    return 0


def plot_predictions_with_uncertainty(Y_true, Y_pred_mean, Y_pred_std, title="Prediction with Uncertainty"):
    """  Plot true values, predicted mean, and shaded uncertainty region (±1 std).
    Parameters:
    - Y_true: array-like, shape (n_samples,)
    - Y_pred_mean: array-like, shape (n_samples,)
    - Y_pred_std: array-like, shape (n_samples,)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    P = 0.2
    n = len(Y_true)
    size = int(n * P)
    rng = np.random.default_rng(seed=42)  # for reproducibility
    indices = rng.choice(n, size=size, replace=False)
    Y_true = Y_true[indices]
    Y_pred_mean = Y_pred_mean[indices]
    Y_pred_std = Y_pred_std[indices]

    Y_true = np.ravel(Y_true)
    Y_pred_mean = np.ravel(Y_pred_mean)
    Y_pred_std = np.ravel(Y_pred_std)

    x = np.arange(len(Y_true))  # use index as x-axis if no specific feature
    x = Y_pred_mean  # Use predicted mean as x-axis

    plt.figure(figsize=(10, 5))
    swedish_blue = '#005293'  # Hex color
    plt.errorbar(x, Y_pred_mean, yerr=Y_pred_std, color='black', fmt='o', mfc='black', alpha=0.8, label='Prediction ±1 Std Dev')
    plt.scatter(x, Y_true, label='True PCE', color='blue', s=30)

    plt.xlabel("Predicted PCE")
    plt.ylabel("Measured PCE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()