
if __name__ == "__main__":
	import numpy as np
	import pandas as PD
	from sklearn.preprocessing import MinMaxScaler, StandardScaler
	from sklearn.model_selection import train_test_split, KFold, GridSearchCV
	from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
	from sklearn.preprocessing import StandardScaler
	from sklearn.pipeline import Pipeline
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

	# --- 2. Create a Hold-Out Test Set ---
	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y,
		test_size=0.2,  # 20% of data held out for testing
		random_state=42 # for reproducible results
		)
	print(f"Training set size: {X_train.shape[0]} samples")
	print(f"Test set size: {X_test.shape[0]} samples")

	scaler = StandardScaler()
	# We use 'model' as a placeholder for the regressor object.
	model = xgb.XGBRegressor(
		objective='reg:squarederror',
		random_state=42
		)

	# pipeline
	pipe = Pipeline([
		('scaler', scaler),
		('model', model)
		])

	# --- Define Hyperparameters for Tuning ---
	# We create a "grid" of hyperparameters to test.
	param_grid = {
	    'model__n_estimators': [50, 100, 150],
	    'model__max_depth': [5, 6, 7],
	    'model__learning_rate': [0.2, 0.3, 0.4]
	    }

	# ---  Cross-Validation and Grid Search ---
	cv_strategy = KFold(n_splits=5, shuffle=True, random_state=82)
	# 'scoring='neg_mean_squared_error'' is used because CV tries to *maximize*
	grid_search = GridSearchCV(
		estimator=pipe,
		param_grid=param_grid,
		cv=cv_strategy,
		scoring='neg_mean_squared_error',
		n_jobs=-1,
		verbose=1
		)

	print("\nHyperparameter Tuning (GridSearchCV)...")
	# It cross-validates all hyperparameter combinations on (X_train, Y_train).
	grid_search.fit(X_train, Y_train)

	# --- Report Tuning Results ---
	print("\n--- Tuning Results ---")
	print(f"Best Hyperparameters Found:")
	print(grid_search.best_params_)

	# 'grid_search.best_score_' is the 'neg_mean_squared_error'. We multiply by -1 to make it positive (standard MSE)
	best_cv_mse = -grid_search.best_score_
	best_cv_rmse = np.sqrt(best_cv_mse)
	print(f"\nBest Cross-Validated RMSE: {best_cv_rmse:.2f}")

	# --- Get the Final Model and Evaluate on Test Set ---
	# The 'grid_search' object automatically re-trains a final model on the *entire* (X_train, Y_train) dataset using the best hyperparameters.
	# final deployable model.
	final_model = grid_search.best_estimator_

	# Now, we use the "locked away" test set for the first and only time.
	print("\n--- Final Model Evaluation (on unseen Test Set) ---")
	Y_pred = final_model.predict(X_test)

	#test_mse = mean_squared_error(Y_test, Y_pred)
	#test_rmse = np.sqrt(test_mse)
	#print(f"Test Set RMSE: {test_rmse:.4f}")

	final_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
	final_mae = mean_absolute_error(Y_test, Y_pred)
	final_mape = mean_absolute_percentage_error(Y_test, Y_pred)
	print(f"Final Test Set RMSE: {final_rmse:.2f}")
	print(f"Final Test Set MAE: {final_mae:.2f}")
	print(f"Final Test Set MAPE: {final_mape:.2f}")

	# --- Save Final Model.	Saves the *entire pipeline* (scaler + model) to a file.
	model_filename = 'final_xgboost_model.joblib'
	joblib.dump(final_model, model_filename)
	print(f"\nFinal model (pipeline) saved to '{model_filename}'")


	# --- 9. How to Load and Use the Model Later ---
	# This is how you would use it in a different script.
	#print("\n--- Example: Loading and Predicting ---")
	#loaded_model = joblib.load(model_filename)
	# Create some new, unseen data (e.g., one sample)
#	new_data = np.random.rand(1, 10)  # Must have 10 features
	# 'loaded_model.predict()' will automatically:
	# 1. Scale the 'new_data' (using the scaler fit on X_train)
	# 2. Make a prediction with the XGBoost model
#	prediction = loaded_model.predict(new_data)
#	print(f"Prediction on new data: {prediction[0]}")