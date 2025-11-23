import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline


def objective_function(synthesis_inputs, material, target_PCE, loaded_model, input_bounds):
    # Cost function that optimizes ONLY the synthesis variables.
    # Args: - synthesis_inputs (np.array): 1D array of 5 synthesis variables.
    #       -  fixed_material_inputs (np.array): 1D array of 4 material variables.
    
    import numpy as np
    #Check if any *synthesis* input is out of bounds
    for i in range(len(synthesis_inputs)):
        if not (input_bounds[i][0] <= synthesis_inputs[i] <= input_bounds[i][1]):
            return 1e9  # A massive penalty

    full_inputs = np.hstack([material, synthesis_inputs])   # Reconstruct the full 9-element input vector
    inputs_reshaped = full_inputs.reshape(1, -1)
    pce_prediction = loaded_model.predict(inputs_reshaped)
    cost = np.abs(target_PCE - pce_prediction[0]) * np.abs(target_PCE - pce_prediction[0])   # Squared Error.
    
    return cost


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
    # Inverse-transform a batch of p-dimensional vectors back to original scale using a StandardScaler.
    n_samples, p = X_target.shape
    N = len(target_dims) + len(observed_dims)

    # Create zero-padded full matrix
    X_full = np.zeros((n_samples, N))
    X_full[:, target_dims] = X_target

    X_full_inv = scaler.inverse_transform(X_full)
    X_target_inv = X_full_inv[:, target_dims]

    return X_target_inv


def inferre_exp_conditions(regressor_file, generator_file, target_PCE, material, input_bounds):
    from pathlib import Path
    import sys
    sys.path.insert(0, './src')  # add src to Python path
    import GMM_utils as GMM
    import ABX

    Nsamples_generator = 2   # number of initial points we take for optimization.
    K = 1                     # number of best inferred inputs.

    try:
        loaded_model = joblib.load(regressor_file)
    except FileNotFoundError:
        print('Error loading the regression model.')

    try:
        gen_model = joblib.load(generator_file)
    except FileNotFoundError:
        print('Error loading the generator model.')

    gmm_model = gen_model['model']
    scaler_X  = gen_model['scaler_x']
    scaler_y  = gen_model['scaler_y']

    # --- To obtain generated synthesis desriptors assuming we have the perovskite material and a desired PCE.
    observed_dims = [0, 1, 2, 3, 9]
    target_dims   = [4, 5, 6, 7, 8]
    #z_material = [-0.036007170669086666, 0.006586621085668454, 0.01751554519205328, -0.0013882326952161786]   # Representation of MAPbI3
    z_material = material
    z_pce      = target_PCE
    Zo = array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y)   #---- conditional input vector.
    cond_gmm = condition_gmm(gmm_model, observed_dims, target_dims, Zo)            # conditional gmm.
    new_samples = sample_from_conditional_gmm(cond_gmm, n_samples=Nsamples_generator, random_state=None)
    observed_dims = [0, 1, 2, 3]       # input observed dims.
    X_gen = inverse_transform_batch(new_samples, scaler_X, target_dims, observed_dims)
    #print(X_gen)    # We have obtained a set of initial points.

    ii = 0
    optimization_results = {}
    for init_vector in X_gen:
        synthesis_inputs = init_vector
        # Run COBYLA optimization: (Powell, 1994)
        initial_step_size = 0.05
        result = minimize(
            objective_function,           # The function to minimize
            synthesis_inputs,             # The 5-element vector to change
            method='COBYLA',
            args=(material, target_PCE, loaded_model, input_bounds), 
            options={'maxiter': 50,
                'rhobeg': initial_step_size,
                'tol': 1e-4,                 #tolerance
                'disp': False
                }
            )
        ## Run the Nelder-Mead optimization : (Nelder and Mead, 1965)
        #result = minimize(
        #    objective_function,           # The function to minimize
        #    synthesis_inputs,             # The 5-element vector to change
        #    method='Nelder-Mead',
        #    args=(material, target_PCE, loaded_model, input_bounds), 
        #    options={'maxiter': 50,
        #        'rhobeg': initial_step_size,
        #        'xatol': 1e-2,
        #        'disp': False
        #        }
        #    )

        if result.success:
            optimal_synthesis_inputs = result.x
            final_cost = result.fun
            # Check the final prediction
            optimal_full_inputs = np.hstack([material, optimal_synthesis_inputs])
            final_pce_prediction = loaded_model.predict(optimal_full_inputs.reshape(1, -1))[0]
            optimization_results[ii] = {
                'status': 'success',
                'optimal_inputs': optimal_full_inputs,
                'predicted_pce': final_pce_prediction,
                'cost': final_cost,
                'initial_guess': init_vector
                }
        else:
            print(f"  Status: FAILED ({result.message})")
            optimization_results[ii] = {
                'status': 'failed',
                'message': result.message,
                'initial_guess': init_vector
                }
        ii = ii + 1

    # Filter out the failures
    successful_runs = {k: v for k, v in optimization_results.items() if v['status'] == 'success'}
    best_candidates = sorted(
        successful_runs.items(), 
        key=lambda item: abs(target_PCE - item[1]['predicted_pce'])
        )

    inverted_points = best_candidates[:K]
    # Create empty lists to store the extracted data
    optimal_inputs_list = []
    predicted_pce_list = []
    for run_index, results in inverted_points:
        if results['status'] == 'success':
            if results['cost'] < 1e9: 
                # Append the data to your new lists
                optimal_inputs_list.append(results['optimal_inputs'])
                predicted_pce_list.append(results['predicted_pce'])

    return optimal_inputs_list







def inferre_by_opt(regressor_file, target_PCE, material, input_bounds):
    from pathlib import Path
    import sys

    Nsamples_generator = 2   # number of initial points we take for optimization.
    K = 1                     # number of best inferred inputs.

    try:
        loaded_model = joblib.load(regressor_file)
    except FileNotFoundError:
        print('Error loading the regression model.')

    # bound the sampled data within the range of trainig data.
    bounds_array = np.array(input_bounds)
    min_vals = bounds_array[:, 0]  # First column
    max_vals = bounds_array[:, 1]  # Second column
    n_rows = Nsamples_generator
    n_cols = len(min_vals)         # Should be 5
    random_array = np.random.uniform(
        low=min_vals,
        high=max_vals, 
        size=(n_rows, n_cols)
        )

    ii = 0
    optimization_results = {}
    for init_vector in random_array:
        synthesis_inputs = init_vector
        # Run COBYLA optimization: (Powell, 1994)
        initial_step_size = 0.05
        result = minimize(
            objective_function,           # The function to minimize
            synthesis_inputs,             # The 5-element vector to change
            method='COBYLA',
            args=(material, target_PCE, loaded_model, input_bounds), 
            options={'maxiter': 50,
                'rhobeg': initial_step_size,
                'tol': 1e-4,                 #tolerance
                'disp': False
                }
            )
        ## Run the Nelder-Mead optimization : (Nelder and Mead, 1965)
        #result = minimize(
        #    objective_function,           # The function to minimize
        #    synthesis_inputs,             # The 5-element vector to change
        #    method='Nelder-Mead',
        #    args=(material, target_PCE, loaded_model, input_bounds), 
        #    options={'maxiter': 50,
        #        'rhobeg': initial_step_size,
        #        'xatol': 1e-2,
        #        'disp': False
        #        }
        #    )


        if result.success:
            optimal_synthesis_inputs = result.x
            final_cost = result.fun
            # Check the final prediction
            optimal_full_inputs = np.hstack([material, optimal_synthesis_inputs])
            final_pce_prediction = loaded_model.predict(optimal_full_inputs.reshape(1, -1))[0]
            optimization_results[ii] = {
                'status': 'success',
                'optimal_inputs': optimal_full_inputs,
                'predicted_pce': final_pce_prediction,
                'cost': final_cost,
                'initial_guess': init_vector
                }
        else:
            print(f"  Status: FAILED ({result.message})")
            optimization_results[ii] = {
                'status': 'failed',
                'message': result.message,
                'initial_guess': init_vector
                }
        ii = ii + 1

    # Filter out the failures
    successful_runs = {k: v for k, v in optimization_results.items() if v['status'] == 'success'}
    best_candidates = sorted(
        successful_runs.items(), 
        key=lambda item: abs(target_PCE - item[1]['predicted_pce'])
        )

    inverted_points = best_candidates[:K]
    # Create empty lists to store the extracted data
    optimal_inputs_list = []
    predicted_pce_list = []
    for run_index, results in inverted_points:
        if results['status'] == 'success':
            if results['cost'] < 1e9: 
                # Append the data to your new lists
                optimal_inputs_list.append(results['optimal_inputs'])
                predicted_pce_list.append(results['predicted_pce'])

    return optimal_inputs_list






def verify_inferred_inputs(regressor_file, material, optimal_synthesis_inputs):
    # Check the final prediction
    try:
        loaded_model = joblib.load(regressor_file)
    except FileNotFoundError:
        print('Error loading the regression model.')
        return None

    synthesis_batch = np.array(optimal_synthesis_inputs)
    if synthesis_batch.size == 0:
        print("Input list is empty.")
        return np.array([]) # Return an empty array

    pce_predictions = loaded_model.predict(synthesis_batch)

    return pce_predictions




if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib

    from pathlib import Path
    import sys
    sys.path.insert(0, './src')  # add src to Python path
    import GMM_utils as GMM
    import ABX

    xgboost_filename = 'final_xgboost_model.joblib'   # the direct model alreadt trained.
    gmm_filename     = 'gmr_model_5_clusters.joblib'

    material = np.array([-0.036007170669086666, 0.006586621085668454, 0.01751554519205328, -0.0013882326952161786])   #Representation of MAPbI3
    input_bounds = [
        [0.01, 5.0],
        [1.0, 5.0],
        [1.1, 2.2],
        [10, 300],
        [0.01, 10]
        ]

    all_targets_list = []
    all_predictions_list = []
    opt_all_targets_list = []
    opt_all_predictions_list = []
    pce_targets = np.arange(10.0, 22.2, 0.05)
    print(f"Starting inverse modeling for {len(pce_targets)} target values...")
    for target_PCE in pce_targets:
        print(f"--- Processing Target PCE: {target_PCE:.1f} ---")
        optimal_inputs_list = inferre_exp_conditions(xgboost_filename, gmm_filename, target_PCE, material, input_bounds)
        optimal_inputs_array = np.array(optimal_inputs_list)
        if not optimal_inputs_list:
            print("  No valid initial conditions found for this target. Skipping.")
            continue

        pce_predictions = verify_inferred_inputs(xgboost_filename, material, optimal_inputs_list) # Get final predicted PCEs
        n_predictions = len(pce_predictions)
        all_predictions_list.extend(pce_predictions)
        all_targets_list.extend([target_PCE] * n_predictions)

        # ----- now the same process, but for random initialization.
        the_list_opt = inferre_by_opt(xgboost_filename, target_PCE, material, input_bounds)
        the_list_opt = np.array(the_list_opt)
        if not optimal_inputs_list:
            print("No valid initial conditions found for this target. Skipping.")
            continue
        opt_pce_predictions = verify_inferred_inputs(xgboost_filename, material, the_list_opt) # Get final predicted PCEs
        opt_n_predictions = len(opt_pce_predictions)
        opt_all_predictions_list.extend(opt_pce_predictions)
        opt_all_targets_list.extend([target_PCE] * opt_n_predictions)


    # RMSE for both methods.
    from sklearn.metrics import root_mean_squared_error
    print('----------------------------')
    rmse_gmm_start_opt = root_mean_squared_error(all_targets_list, all_predictions_list)
    rmse_rand_start_opt  = root_mean_squared_error(opt_all_targets_list, opt_all_predictions_list)
    print('RMSE of Random-Start Optimization method is: ')
    print(rmse_rand_start_opt)
    print('RMSE of GMM-assisted Optimization method is: ')
    print(rmse_gmm_start_opt)
    print('----------------------------')

    print("\n--- All targets processed. Generating plot. ---")
    # ---Create the Scatter Plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(opt_all_targets_list, opt_all_predictions_list, c='black', alpha=0.3, label="random-start optimization", marker='s')
    plt.scatter(all_targets_list, all_predictions_list, c='#0047AB', alpha=1.0, label="GMM-assisted optimization")
    max_val = max(all_targets_list + all_predictions_list)
    min_val = min(all_targets_list + all_predictions_list)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="perfect coincidence")
    plt.xlabel("target PCE", fontsize=14)
    plt.ylabel("predicted PCE", fontsize=14)
    plt.title("Inverse modeling validation", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':')
    plt.axis('equal') # Ensures the x and y axes have the same scale
    plt.tight_layout()
    plt.show()


    #-------------------------------------------------------
    