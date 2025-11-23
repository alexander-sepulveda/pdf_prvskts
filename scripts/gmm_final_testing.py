# We have already selected the number of components; thus, we
# train with the whole training dataset; and, we validate with
# the 20% tet set.


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

	N_CLUSTERS = 5

	# === TRAIN THE FINAL MODEL ===
	print("\nTraining final model on the *entire* 'train_full' dataset...")

	# 1. Create final scalers, fit on ALL training data
	final_scaler_x = StandardScaler().fit(X_train_full)
	final_scaler_y = StandardScaler().fit(Y_train_full)

	# 2. Scale all training data
	X_train_full_s = final_scaler_x.transform(X_train_full)
	Y_train_full_s = final_scaler_y.transform(Y_train_full)

	# 3. Concatenate scaled data
	final_data_s = np.hstack((X_train_full_s, Y_train_full_s))

	# 4. Train the final model
	final_gmr_model = GMM.GaussianMixtureRegression(N_CLUSTERS, 'full', random_state=42)

	final_gmr_model.fit(final_data_s, n_inputs, n_outputs)
	print("Final model is trained.")

	# FINAL EVALUATION (on the 'test' set)
	print("\n--- Final Model Evaluation (on unseen Test Set) ---")

	
	X_test_s = final_scaler_x.transform(X_test)    # Scale the UNSEEN X_test data
	Y_pred_s, Y_std_s, _ = final_gmr_model.predict(X_test_s)   # Predict in the scaled space
	Y_pred_final = final_scaler_y.inverse_transform(Y_pred_s)  # Inverse-transform the prediction back to the original space
	std_y_scale = final_scaler_y.scale_ if hasattr(final_scaler_y, 'scale_') else 1.0
	Y_std_final = Y_std_s * std_y_scale
	# Calculate final performance metrics
	final_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_final))
	final_mae = mean_absolute_error(Y_test, Y_pred_final)
	final_mape = mean_absolute_percentage_error(Y_test, Y_pred_final)

	# -- now, we separate the prediction in two levels: high and low; where, we use
	# the threshold for high as in 17% in PCE; in accordance with few recent articles.
	condicion_high = Y_test >= 17
	Y_pred_final_high = Y_pred_final[condicion_high]
	Y_test_high = Y_test[condicion_high]
	rmse_high = np.sqrt(mean_squared_error(Y_test_high, Y_pred_final_high))
	mae_high  = mean_absolute_error(Y_test_high, Y_pred_final_high)
	mape_high = mean_absolute_percentage_error(Y_test_high, Y_pred_final_high)
	print('---------------------------------')
	print("Result for PCE>17%:")
	print(f"mae high: {mae_high:.2f}")
	print(f"rmse high: {rmse_high:.2f}")
	print(f"mape high: {mape_high:.2f}")
	print('-----')

	condicion_low = Y_test < 17
	Y_pred_final_low = Y_pred_final[condicion_low]
	Y_test_low  = Y_test[condicion_low]
	rmse_low  = np.sqrt(mean_squared_error(Y_test_low, Y_pred_final_low))
	mae_low   = mean_absolute_error(Y_test_low, Y_pred_final_low)
	mape_low  = mean_absolute_percentage_error(Y_test_low, Y_pred_final_low)
	print("Result for PCE>17%:")
	print(f"mae low: {mae_low:.2f}")
	print(f"rmse low: {rmse_low:.2f}")
	print(f"mape low: {mape_low:.2f}")
	print('----------------------------------')
	# ------------------------------------------------------------------------


	print(f"Final Test Set MAE: {final_mae:.2f}")
	print(f"Final Test Set RMSE: {final_rmse:.2f}")
	print(f"Final Test Set MAPE: {final_mape:.2f}")

	# === SAVE THE DEPLOYABLE MODEL ===
	model_package = {
	    'model': final_gmr_model,
	    'scaler_x': final_scaler_x,
	    'scaler_y': final_scaler_y
	    }
	#file_name = base_dir / f"gmr_model_{N_CLUSTERS}_clusters.joblib"
	#joblib.dump(model_package, f"gmr_model_{N_CLUSTERS}_clusters.joblib") # Save gmm model
	#joblib.dump(X_test, 'X_test.joblib')                         # Save testing variable
	#joblib.dump(Y_test, 'Y_test.joblib')                         # Save testing variable
	#joblib.dump(X_train_full, 'X_train_full.joblib')             # train+test (cross-validation)
	#joblib.dump(Y_train_full, 'Y_train_full.joblib')             # train+test (cross-validation)
	#print(f"\nFinal model package saved as 'gmr_model_{N_CLUSTERS}_clusters.joblib'")