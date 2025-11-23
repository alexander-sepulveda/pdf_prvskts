import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture

from pathlib import Path
base_dir = Path(__file__).parent.parent    # project location
import sys
sys.path.insert(0, './src')  # add src to Python path
import plot_util
import ABX
import GMM_utils as GMM


def get_original_scale_centers(model_path, n_input_features):
    import pandas as pd
    """
    Loads the GMM package and transforms the cluster centers
    back to their original data scale.
    """
    
    # 1. Load the entire model package
    try:
        package = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Creating a dummy model file to run this example...")
        create_dummy_model_file(model_path, n_input_features)
        package = joblib.load(model_path)
    
    # 2. Extract all components from the package
    gmm_model = package['model']
    scaler_x = package['scaler_x']
    scaler_y = package['scaler_y']

    # 3. Get the centers from the fitted GMM
    # These centers are in the *scaled* (Z-score) space
    centers_scaled = gmm_model.gmm.means_
    n_clusters = gmm_model.gmm.n_components
    
    print(f"\nLoaded {n_clusters} centers from the GMM.")
    print(f"Shape of scaled centers: {centers_scaled.shape}")

    # 4. Split the centers back into their input (X) and output (Y) parts
    # The first 'n_input_features' (e.g., 9) columns are inputs
    centers_scaled_x = centers_scaled[:, :n_input_features]
    
    # The remaining columns are outputs
    centers_scaled_y = centers_scaled[:, n_input_features:]

    # 5. Apply the inverse_transform using the correct scaler for each part
    centers_original_x = scaler_x.inverse_transform(centers_scaled_x)
    centers_original_y = scaler_y.inverse_transform(centers_scaled_y)

    # 6. Recombine the parts to get the final, original-scale centers
    centers_original = np.hstack((centers_original_x, centers_original_y))
    
    return centers_original


def load_and_get_centers(model_path):
    try:
        package = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Creating a dummy model file to run this example...")
        create_dummy_model_file()
        package = joblib.load(model_path)
    
    gmm_model = package['model']
    centers_zscore_space = gmm_model.gmm.means_
    weights = gmm_model.gmm.weights_
    
    return centers_zscore_space, gmm_model.gmm.n_components, weights

def plot_radar_chart(centers, feature_names, weights):
    """
    Generates a radar chart from the cluster centers.
    """
    num_clusters, num_features = centers.shape

    performance_labels = [
        "Top",
        "High",
        "Moderate",
        "Low",
        "Poor"
        ]
    if num_clusters != len(performance_labels):
        print(f"Warning: You have {num_clusters} clusters but {len(performance_labels)} labels.")
        performance_labels = [f"Cluster {i}" for i in range(num_clusters)]

    # Scale centers to a 0-1 range for plotting
    plot_scaler = MinMaxScaler()
    centers_for_plot = plot_scaler.fit_transform(centers)

    # Create the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # -- DEFINE YOUR CUSTOM COLOR LIST ---
    colors = [
        '#0000ff',  # blue
        '#00FFFF',  # cyan
        '#808000',  # Cooked Asparagus Green  2ca02c
        '#ff7f0e',  # Safety Orange
        '#d62728',  # Brick Red
        '#9467bd',  # Muted Purple
        '#8c564b',  # Chestnut Brown
        '#e377c2',  # Barbie Pink
    ]
    
    # Plot each cluster
    for i in range(num_clusters):
        values = centers_for_plot[i].tolist()
        values += values[:1]  # Close the loop

        name = performance_labels[i]
        label_text = f"{name} (weight: {weights[i]*100:.1f}%)"   # Format the final label text
        color = colors[i % len(colors)]     # Get and apply the custom color
        #    This highlights the first cluster (i == 0).
        if i == 1:  # This will be your "champion" cluster
            linewidth = 3.0   # Thicker line
            fill_alpha = 0.35 # More opaque fill
            plot_zorder = 10  # Draw on top
        else:
            linewidth = 2.0   # Standard line
            fill_alpha = 0.2  # Standard fill
            plot_zorder = 5   # Draw behind
        
        ax.plot(angles, values, linewidth=linewidth, linestyle='solid', 
                label=label_text, color=color, zorder=plot_zorder)
        ax.fill(angles, values, alpha=fill_alpha, color=color)

    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=13)
    #ax.set_title("cluster center profiles", size=14, y=1.1)
    ax.set_rlim(0, 1.0)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13)
    print("\nRadar chart has been generated.")
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # 1. Define model path and feature names
    MODEL_FILE = "gmr_model_5_clusters.joblib"


    FEATURE_NAMES = ['L$_1$', 'L$_2$', 'L$_3$', 'L$_4$', '$\chi_{sol.}$ (solvents)', 'TB (thermal budget)', '$E_g$ (band gap)', '$T_1$ ($1_{st}$ temperature)', '$\mathcal{A}$ (measured area)', 'PCE']

    cluster_centers, n_clusters, cluster_weights = load_and_get_centers(MODEL_FILE)
    descend_ind = np.argsort(cluster_centers[:, -1])[::-1]
    centers = cluster_centers[descend_ind]
    weights = cluster_weights[descend_ind]

    n_clusters = len(centers)    #Update the final cluster count for the plot

    N_INPUTS = len(FEATURE_NAMES)-1
    original_centers = get_original_scale_centers(MODEL_FILE, N_INPUTS)
    original_centers = original_centers[descend_ind]
    # Print the results in a clean DataFrame
    print("\n--- GMM Cluster Centers (Original Scale) ---")
    if len(FEATURE_NAMES) != original_centers.shape[1]:
        print("Warning: Feature names list length doesn't match center dimensions.")
        print(original_centers)
    else:
        df_centers = pd.DataFrame(original_centers, columns=FEATURE_NAMES)
        df_centers['$\chi_{sol.}$ (solvents)'] = np.power(10, df_centers['$\chi_{sol.}$ (solvents)'])
        df_centers['TB (thermal budget)'] = np.power(10, df_centers['TB (thermal budget)'])

        df_centers['$\chi_{sol.}$ (solvents)'] = df_centers['$\chi_{sol.}$ (solvents)'].round(2)
        df_centers['TB (thermal budget)'] = df_centers['TB (thermal budget)'].round(2)
        df_centers['$E_g$ (band gap)'] = df_centers['$E_g$ (band gap)'].round(2)
        df_centers['$T_1$ ($1_{st}$ temperature)'] = df_centers['$T_1$ ($1_{st}$ temperature)'].round(2)
        df_centers['$\mathcal{A}$ (measured area)'] = df_centers['$\mathcal{A}$ (measured area)'].round(2)
        df_centers['PCE'] = df_centers['PCE'].round(2)

        print(df_centers.to_string()) # .to_string() ensures all columns print


    # Create the plot
    if len(FEATURE_NAMES) != centers.shape[1]:
        print(f"\n--- ERROR ---")
        print(f"Feature name mismatch: Your FEATURE_NAMES list has {len(FEATURE_NAMES)} items,")
        print(f"but your model centers have {centers.shape[1]} features.")
    else:
        plot_radar_chart(centers, FEATURE_NAMES, weights)
