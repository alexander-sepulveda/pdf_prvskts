def plot_tsne_comparison(data_test, data_gmm, data_gan, perplexity=30, random_state=42):
    #    Combines, projects, and plots a t-SNE comparison of three datasets.
    # Stack all data into one large array
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    combined_data = np.vstack([data_test, data_gmm, data_gan])
    
    # Create corresponding labels
    labels_test = np.full(data_test.shape[0], 'Real Data')
    labels_gmm = np.full(data_gmm.shape[0], 'GMM')
    labels_gan = np.full(data_gan.shape[0], 'GAN')
    combined_labels = np.concatenate([labels_test, labels_gmm, labels_gan])

    print(f"Combined data shape: {combined_data.shape}")

    # --- 2. (Pro-Tip) Reduce Dimensions with PCA first ---
    # t-SNE is slow. Running PCA first speeds it up and reduces noise.
    # We reduce to 50 dimensions (or fewer if your data has < 50 features)
    n_pca_components = min(50, combined_data.shape[1])
    print(f"Running PCA (reducing to {n_pca_components} components)...")
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    combined_data_pca = pca.fit_transform(combined_data)

    # --- 3. Run t-SNE ---
    print(f"Running t-SNE (perplexity={perplexity})... This may take a minute.")
    tsne = TSNE(n_components=2, 
                perplexity=perplexity, 
                random_state=random_state, 
                max_iter=1000,
                init='pca', # Use PCA initialization
                n_jobs=-1) # Use all available CPU cores
    
    tsne_results = tsne.fit_transform(combined_data_pca)

    # --- 4. Create a DataFrame for Plotting ---
    df = pd.DataFrame()
    df['tsne-1'] = tsne_results[:, 0]
    df['tsne-2'] = tsne_results[:, 1]
    df['label'] = combined_labels

    # --- 5. Plot with Seaborn ---
    print("Plotting results...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='tsne-1',
        y='tsne-2',
        hue='label',           # Automatically colors by this column
        alpha=0.9,             # Set transparency
        s=40,                  # Set marker size
        style='label',         # Use different markers
        palette='dark'         # Set color palette
    )
    
    plt.title(f't-SNE Comparison (Perplexity: {perplexity})', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    return df



def ksdensity_4_dimensions(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Plot t-SNE Dimension 1
    sns.kdeplot(
        data=df,
        x='tsne-1',
        hue='label',         # separate line for each label
        fill=False,          # no area under the curve
        alpha=0.6,           # Sets transparency
        common_norm=False,   # curves normalized to 1
        ax=axes[0]           
        )
    axes[0].set_title('KDE Comparison for t-SNE Dimension 1', fontsize=14)
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].grid(True, linestyle=':')

    # Plot t-SNE Dimension 2
    sns.kdeplot(
        data=df,
        x='tsne-2',
        hue='label',
        fill=True,
        alpha=0.6,
        common_norm=False,
        ax=axes[1]
        )
    axes[1].set_title('KDE Comparison for t-SNE Dimension 2', fontsize=14)
    axes[1].set_xlabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].grid(True, linestyle=':')

    fig.suptitle('Comparison of t-SNE Component Distributions', fontsize=18, y=1.03)
    plt.tight_layout()
    plt.show()







# sample with full pdf model.
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
base_dir = Path(__file__).parent.parent    # project location
import sys
sys.path.insert(0, './src')  # add src to Python path
import plot_util
import ABX
import GMM_utils as GMM


# Load the saved model and scalers
model_package = joblib.load('gmr_model_5_clusters.joblib')
gmr_model = model_package['model']
scaler_x  = model_package['scaler_x']
scaler_y  =  model_package['scaler_y']

## Load the train data
#try:
#    X_train = joblib.load('X_train_full.joblib')
#    Y_train = joblib.load('Y_train_full.joblib')
#except FileNotFoundError:
#    print("Error: some of the train files not found.")
#data_train = np.column_stack((X_train, Y_train))

# ---------------------------------------------
# Load the saved test data
try:
    X_test = joblib.load('X_test.joblib')
    Y_test = joblib.load('Y_test.joblib')
except FileNotFoundError:
    print("Error: 'X_test.joblib' or 'Y_test.joblib' not found.")
data_test = np.column_stack((X_test, Y_test))
N_sample_gen = len(X_test)            # the number of samples of the test set
print('number of test samples:')
print(N_sample_gen)
print('                              ')



# --------- create a gmm clone for sampling data.
from sklearn.mixture import GaussianMixture
gmm_sampler = GaussianMixture(
    n_components = gmr_model.gmm.n_components,
    covariance_type = gmr_model.gmm.covariance_type,
    random_state=None  # <-- unlocks the sampler
    )
gmm_sampler.weights_ = gmr_model.gmm.weights_
gmm_sampler.means_ = gmr_model.gmm.means_
gmm_sampler.covariances_ = gmr_model.gmm.covariances_
gmm_sampler.precisions_cholesky_ = gmr_model.gmm.precisions_cholesky_

# --- Generate New Samples from the GMM ---
n_inputs  = scaler_x.n_features_in_   # Get dimensions
n_outputs = scaler_y.n_features_in_
gmm_sampler.random_state = 42         # Set the state
joint_samples_scaled, _ = gmm_sampler.sample(N_sample_gen)
X_samples_scaled = joint_samples_scaled[:, :n_inputs]       # Split the scaled samples into X and Y
Y_samples_scaled = joint_samples_scaled[:, n_inputs:]
X_samples = scaler_x.inverse_transform(X_samples_scaled)    # unscale the data (back to the original scale)
Y_samples = scaler_y.inverse_transform(Y_samples_scaled)
data_gmm  = np.column_stack((X_samples, Y_samples))



# ---------------------------------------------
# -- energy distance test of homogeneity
# H_o : the two random vectors have the same distribution.
# H_1 : they have different distributions.
import dcor   # https://www-sciencedirect-com.translate.goog/science/article/pii/S2352711023000225?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
dtest_result = dcor.homogeneity.energy_test(data_test, data_gmm, num_resamples=99, exponent=1, random_state=42)
print('---------------------')
print('hypothesis test in respect to gmm data sampled.')
print(dtest_result)

data_gan = np.loadtxt('generated_perovskite_GAN3.csv', delimiter=',', skiprows=1)
print('---------------------')
print('hypothesis test in respect to GAN data sampled.')
dtest_result = dcor.homogeneity.energy_test(data_test, data_gan, num_resamples=99, exponent=1, random_state=42)
print(dtest_result)


# ----------------------------------------
# ---- Maximum Mean Discrepancy two-sample test --
# H_o : the two random vectors have the same distribution.
# H_1 : they have different distributions.
#from hyppo.ksample import MMD
#mmd_test = MMD(num_permutations=999, random_state=42)
#mmd_result = mmd_test.test(data_test, data_gmm)
#print('---------------------')
#print('')
#print('hypothesis test in respect to gmm data sampled.')
#print(mmd_result)
#print('---------------------')
#print('hypothesis test in respect to GAN data sampled.')
#mmd_result = mmd_test.test(data_test, data_gan)
#print(mmd_result)



#plot_tsne_comparison(data_test, data_gmm, data_gan, perplexity=20, random_state=42)





















#joint_samples_scaled, _ = gmm_sampler.sample(N_sample_gen)

#p_value_list = []
#print("Running repeated E-distance tests...")
#for i in range(1):
#    gmm_sampler.random_state = 42 + i # <-- Set the state for this loop
#    #joint_samples_scaled, _ = gmm_sampler.sample(N_sample_gen)
#    joint_samples_scaled, _ = gmm_sampler.sample(4300)
#    X_samples_scaled = joint_samples_scaled[:, :n_inputs]       # Split the scaled samples into X and Y
#    Y_samples_scaled = joint_samples_scaled[:, n_inputs:]
#    X_samples = scaler_x.inverse_transform(X_samples_scaled)    # unscale the data (back to the original scale)
#    Y_samples = scaler_y.inverse_transform(Y_samples_scaled)
#    data_gmm = np.column_stack((X_samples, Y_samples))

#    #-- bound the sampled data within the range of trainig data.
#    min_vals = data_train.min(axis=0)     #min/max range for each column
#    max_vals = data_train.max(axis=0)
#    is_above_min = data_gmm >= min_vals    #boolean "in-bounds" mask
#    is_below_max = data_gmm <= max_vals
#    in_bounds_mask = (is_above_min) & (is_below_max)   #combine masks
#    row_mask = in_bounds_mask.all(axis=1)
#    data_gmm = data_gmm[row_mask]

#    dtest_result = dcor.homogeneity.energy_test(data_test, data_gmm, num_resamples=99, exponent=1, random_state=42)
#    print(dtest_result)
#    p_value_list.append(dtest_result.pvalue)

#print("\n--- P-Value Distribution ---")
#print(p_value_list)
#print(f"Average p-value: {np.mean(p_value_list)}")


# we apply t-SNE to three categoeries real data, GMM gnerated data and GAN generated data.
#data_gan = np.loadtxt('generated_perovskite_GAN3.csv', delimiter=',', skiprows=1)

#print('*********************')
#dtest_result = dcor.homogeneity.energy_test(data_test, data_gan, num_resamples=99, exponent=1, random_state=42)
#print(dtest_result)

#data_test
#data_gmm
#data_gan

#df = plot_tsne_comparison(data_train, data_gmm, data_gan, perplexity=30, random_state=42)