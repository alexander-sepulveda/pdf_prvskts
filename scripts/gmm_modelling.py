def apply_1se_rule(cv_results, n_folds, complexity_key='N_mix'):
	# Applies the 1-Standard-Error (1-SE) rule to select the simplest model.
    print("\n--- Applying 1-SE Rule ---")
    
    # Convert cv_results to a list of dictionaries
    results_list = []
    for params_str, scores in cv_results.items():
        params_dict = eval(params_str)  # Convert string back to dict
        params_dict['mean_mae'] = scores['mean']
        params_dict['std_mae_folds'] = scores['std'] # This is the SD of the folds
        params_dict['params_str'] = params_str
        results_list.append(params_dict)

    # Sort the list by complexity (simplest model first)
    results_list.sort(key=lambda x: x[complexity_key])

    # Find the model with the absolute minimum error
    best_model = min(results_list, key=lambda x: x['mean_mae'])
    min_mae = best_model['mean_mae']
    std_of_best_model_folds = best_model['std_mae_folds']
    
    # Calculate the Standard Error of the Mean (SEM)
    se_of_best_model = std_of_best_model_folds / np.sqrt(n_folds)

    threshold = min_mae + se_of_best_model        # Calculate the 1-SE threshold
    # Find the *simplest* model whose mean MAE is below the threshold
    for model in results_list:
        if model['mean_mae'] < threshold:
            selected_model_stats = model       # This is the simplest model that is "good enough"
            # Re-create the original hyperparameter dict
            best_params = {k: v for k, v in model.items() if k not in ['mean_mae', 'std_mae_folds', 'params_str']}
            print(f"Selected Model (1-SE): {best_params}, MAE={model['mean_mae']:.4f}")
            return best_params, selected_model_stats

    # Fallback in case something goes wrong
    print("Warning: 1-SE rule failed to find a model. Returning absolute best.")
    best_params = {k: v for k, v in best_model.items() if k not in ['mean_mae', 'std_mae_folds', 'params_str']}
    return best_params, best_model



if __name__ == "__main__":
	import numpy as np
	import pandas as PD
	from sklearn.preprocessing import MinMaxScaler, StandardScaler
	from sklearn.model_selection import train_test_split, KFold, GridSearchCV
	from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
	import joblib  # for saving and loading models
	import xgboost as xgb
	from xgboost import XGBRegressor
	from sklearn.decomposition import PCA  # <-- 1. Import PCA
	import itertools
	from pathlib import Path
	base_dir = Path(__file__).parent.parent    # project location
	import sys
	sys.path.insert(0, './src')  # add src to Python path
	import plot_util
	import ABX
	import GMM_utils as GMM

	# we do not use all input variables. When estimating PDFs, the dimension is important.
	var_inputs = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF_DMSO_ratio', 'Perovskite_annealing_thermal_exposure', 'Perovskite_band_gap', 'first_Prvskt_annealing_temperature', 'Cell_area_measured']
	var_outputs = ['JV_default_PCE']
	n_inputs = len(var_inputs)
	n_outputs = len(var_outputs)

	Data = ABX.read_prvskt_data()

	Inputs = Data[var_inputs]
	Outputs = Data[var_outputs]

	X = Inputs.to_numpy()
	Y = Outputs.to_numpy()

	# split the data. (Train+val) + test
	X_train_full, X_test, Y_train_full, Y_test = train_test_split(
		X, Y, test_size=0.2, random_state=42
		)

	# --- Define hyperparameter grid ---
	param_grid = {
	    'N_mix': [2, 3, 4, 5, 6, 7, 8, 9],  # Test multiple values
	    'covariance_type': ['full']
	    }
	cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)

	# Create a list of all parameter combinations
	param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

	cv_results_mae = {} # To store the mean scores for each param combo
	cv_results_mse = {}
	cv_results_mape = {}
	print(f"Starting CV to test {len(param_combinations)} param combinations...")

	for params in param_combinations:
		params_str = str(params)
		fold_scores = []
		folds_mae = []
		folds_mse = []
		folds_mape = []
		# Run k-fold CV *only on the training data*
		k_fold = 1
		for train_index, val_index in cv_strategy.split(X_train_full):
			X_train, X_val = X_train_full[train_index], X_train_full[val_index]
			Y_train, Y_val = Y_train_full[train_index], Y_train_full[val_index]

			# --- Scaling (Correctly inside the loop) ---
			scaler_x = StandardScaler().fit(X_train)
			scaler_y = StandardScaler().fit(Y_train)

			X_train_s = scaler_x.transform(X_train)
			Y_train_s = scaler_y.transform(Y_train)
			X_val_s = scaler_x.transform(X_val)

			# --- Fit GMM (as in your code) ---
			data_s = np.hstack((X_train_s, Y_train_s))
			gmr = GMM.GaussianMixtureRegression(
				params['N_mix'], 
				params['covariance_type'],
				random_state=42
				)

			gmr.fit(data_s, n_inputs, n_outputs)
			
			#--- we save the model for each fold to be later on be used estimate uncertainty.
			# -- We already select J=5 as the number of components.
			#if params['N_mix']==4:
			#	model_fold = {
			#	    'model': gmr,
			#	    'scaler_x': scaler_x,
			#	    'scaler_y': scaler_y
			#	    }
			#	file_name_temp = base_dir / "temp" / f"gmm_cv_{k_fold}.joblib"
			#	joblib.dump(gmr, file_name_temp)
			#	k_fold = k_fold + 1

			# --- Predict and Evaluate ---
			Y_pred_s, _, _ = gmr.predict(X_val_s)
			Y_pred = scaler_y.inverse_transform(Y_pred_s)
			mae_value = mean_absolute_error(Y_val, Y_pred)
			mse_value = np.sqrt(mean_squared_error(Y_val, Y_pred))
			mape_value = mean_absolute_percentage_error(Y_val, Y_pred)
			# mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
			folds_mae.append(mae_value)
			folds_mse.append(mse_value)
			folds_mape.append(mape_value)

		# Store the mean and std dev of the scores across all 10 folds
		cv_results_mae[params_str] = {
			'mean': np.mean(folds_mae),
			'std': np.std(folds_mae)
			}
		cv_results_mse[params_str] = {
			'mean': np.mean(folds_mse),
			'std': np.std(folds_mse)
			}
		cv_results_mape[params_str] = {
			'mean': np.mean(folds_mape),
			'std': np.std(folds_mape)
			}

	# --- Use 1-SE rule to find best hyperparameters ---
	best_params_dict, selected_model_stats = apply_1se_rule(cv_results_mae, n_folds=cv_strategy.get_n_splits(),complexity_key='N_mix')
	best_params_str = str(best_params_dict)

	mae_stats = cv_results_mae[best_params_str]
	mse_stats = cv_results_mse[best_params_str]
	mape_stats = cv_results_mape[best_params_str]
	print(f"  MAE (selection metric): {mae_stats['mean']:.4f} +/- {mae_stats['std']:.4f}")
	print(f"  RMSE (for reference):    {mse_stats['mean']:.4f} +/- {mse_stats['std']:.4f}")
	print(f"  MAPE (for reference):   {mape_stats['mean']:.4f} +/- {mape_stats['std']:.4f}")
