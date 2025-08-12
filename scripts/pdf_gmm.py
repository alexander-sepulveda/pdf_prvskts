



if __name__ == "__main__":
	import numpy as np
	import pandas as PD
	from sklearn.preprocessing import MinMaxScaler, StandardScaler
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import KFold
	from sklearn.metrics import mean_absolute_error
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import mean_absolute_percentage_error
	import joblib  # for saving and loading models
	import xgboost as xgb
	from xgboost import XGBRegressor
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

	Data = ABX.read_prvskt_data()

	Inputs = Data[var_inputs]
	Outputs = Data[var_outputs]

	X = Inputs.to_numpy()
	Y = Outputs.to_numpy()

	#plot the pdfs of synthesis input variables and PCE.
	plot_util.plot_pdfs_inputs(Data)

	# ---- hyparameters.
	N_mix = 7                  # number of multidimensional gaussians.
	covariance_type = 'full'   # You can choose 'full', 'diagonal', 'spherical', or 'tied'

	# 10-fold cross-validation.
	mse_vec = []
	mae_vec = []
	mape_vec = []
	mse_vec_xgboost = []
	mae_vec_xgboost = []
	mape_vec_xgboost = []
	N_sets = 10
	kf = KFold(n_splits=N_sets, shuffle=True, random_state=82)
	for fold, (train_index, test_index) in enumerate(kf.split(X)):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		scaler_standard = StandardScaler()
		X_train = scaler_standard.fit_transform(X_train)
		scaler_y = StandardScaler()
		Y_train = scaler_y.fit_transform(Y_train)

		# Initialize and train the Gaussian Mixture Regression model
		n_inputs = len(var_inputs)
		n_outputs = len(var_outputs)
		data = np.hstack((X_train, Y_train))
		gmr = GMM.GaussianMixtureRegression(N_mix, covariance_type)
		gmr.fit(data, n_inputs, n_outputs)   # Fit model

		# -- now using the XGBoost -------
		model_xgboost = XGBRegressor()
		model_xgboost.fit(X_train, Y_train)

		# Make predictions on the test set
		X_test = scaler_standard.transform(X_test)
		Y_pred, Y_std, Y_pred_map = gmr.predict(X_test)
		# now with XGBoost.
		Y_pred_xgboost = model_xgboost.predict(X_test)
		Y_pred = scaler_y.inverse_transform(Y_pred)
		mse_i = np.sqrt(mean_squared_error(Y_test, Y_pred))
		mse_vec.append(mse_i)
		mae_i = mean_absolute_error(Y_test, Y_pred)
		mae_vec.append(mae_i)
		mape_i = mean_absolute_percentage_error(Y_test, Y_pred)
		mape_vec.append(mape_i)

		# now with XGBoost.
		Y_pred_xgboost = scaler_y.inverse_transform(Y_pred_xgboost.reshape(-1, 1))
		mse_i_xgboost = np.sqrt(mean_squared_error(Y_test, Y_pred_xgboost))
		mse_vec_xgboost.append(mse_i_xgboost)
		mae_i_xgboost = mean_absolute_error(Y_test, Y_pred_xgboost)
		mae_vec_xgboost.append(mae_i_xgboost)
		mape_i_xgboost = mean_absolute_percentage_error(Y_test, Y_pred_xgboost)
		mape_vec_xgboost.append(mape_i_xgboost)

		# Save model
		file_name_temp = base_dir / "temp" / f"gmm_cv_fold{fold+1}.joblib"
		joblib.dump(gmr, file_name_temp)
		vectors_result = {"Y_pred": Y_pred, "Y_std": Y_std, "Y_test": Y_test}
		file_name_temp = base_dir / "temp" / f"cv_results_fold{fold+1}.pkl"
		#joblib.dump(vectors_result, f"cv_results_fold{fold+1}.pkl")
		joblib.dump(vectors_result, file_name_temp)
		file_name_temp = base_dir / "temp" / f"test_indexes{fold+1}.joblib"
		joblib.dump(test_index, file_name_temp)
		file_name_temp = base_dir / "temp" / f"train_indexes{fold+1}.joblib"
		joblib.dump(train_index, file_name_temp)
		print(f"Fold {fold+1}:")

	# ---- performance of GMMs.
	print('performance of GMMs:')
	mean_rmse = np.round(np.mean(mse_vec), 2)
	SE_rmse = np.std(mse_vec) / np.sqrt(N_sets)
	SE_rmse = np.round(SE_rmse, 3)
	print(f"RMSE: {mean_rmse}±{2*SE_rmse}")
	mean_mae = np.round(np.mean(mae_vec), 2)
	SE_mae = np.std(mae_vec) / np.sqrt(N_sets)
	SE_mae = np.round(SE_mae, 3)
	print(f"MAE: {mean_mae}±{2*SE_mae}")
	mean_mape = np.round(np.mean(mape_vec), 2)
	SE_mape = np.std(mape_vec) / np.sqrt(N_sets)
	SE_mape = np.round(SE_mape, 3)
	print(f"MAPE: {mean_mape}±{2*SE_mape}")

	# ---- performance of XGBoost.
	print('performance of XGBoost:')
	mean_rmse = np.round(np.mean(mse_vec_xgboost), 2)
	SE_rmse = np.std(mse_vec_xgboost) / np.sqrt(N_sets)
	SE_rmse = np.round(SE_rmse, 3)
	print(f"RMSE: {mean_rmse}±{2*SE_rmse}")
	mean_mae = np.round(np.mean(mae_vec_xgboost), 2)
	SE_mae = np.std(mae_vec_xgboost) / np.sqrt(N_sets)
	SE_mae = np.round(SE_mae, 3)
	print(f"MAE: {mean_mae}±{2*SE_mae}")
	mean_mape = np.round(np.mean(mape_vec_xgboost), 2)
	SE_mape = np.std(mape_vec_xgboost) / np.sqrt(N_sets)
	SE_mape = np.round(SE_mape, 3)
	print(f"MAPE: {mean_mape}±{2*SE_mape}")

	# Now, we find model with performance closest to MAE average.
	closest_index = np.argmin(np.abs(np.array(mae_vec) - np.mean(mae_vec)))
	print('****** the selected model is :')
	print(base_dir / "temp" / f'gmm_cv_fold{closest_index+1}.joblib')

	loaded = joblib.load(base_dir / "temp" / f"cv_results_fold{closest_index+1}.pkl")
	Y_pred_loaded = loaded["Y_pred"]
	Y_std_loaded  = loaded["Y_std"]
	Y_test_loaded = loaded["Y_test"]

	# now, we plot example predictions with uncertanty.
	std_y = scaler_y.scale_  # shape: (output_dim,)
	print('---------')
	print(Y_std.shape)
	print(std_y.shape)
	std_orig = Y_std * std_y  # broadcasted multiplication
	plot_util.plot_predictions_with_uncertainty(Y_test, Y_pred, std_orig)