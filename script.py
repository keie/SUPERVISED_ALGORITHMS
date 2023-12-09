
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import uniform, randint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import time



def execute_RandomForestRegressor():
    print('--RandomForestRegressor Algorithm starting--')
    start_time = time.time() #tracking start!

    # Load training data and exclude the 'id' column.
    X_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_train.csv').drop('id', axis=1)
    y_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_train.csv')['score']

    # Load the test dataset and save the 'id'.
    X_test = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_test.csv')
    test_ids = X_test['id']
    X_test = X_test.drop('id', axis=1)

    # Split the training data into a training set and a validation set.
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Instantiate the model, in this case, a Random Forest.
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model with the training data.
    model.fit(X_train_split, y_train_split)

    # Make predictions on the validation set to evaluate the model.
    val_predictions = model.predict(X_val_split)

    # Calculate the root mean squared error (RMSE) to assess performance.
    rmse = sqrt(mean_squared_error(y_val_split, val_predictions))

    # If you are satisfied with the model's performance, train it with all the training data (excluding 'id').
    model.fit(X_train, y_train)

    # Make predictions on the test dataset (excluding 'id').
    test_predictions = model.predict(X_test)

    # Create a DataFrame with the IDs and predictions.
    predictions_with_ids = pd.DataFrame({
        'id': test_ids,
        'score': test_predictions
    })
    end_time = time.time()  # Finaliza el seguimiento del tiempo.
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    # Save the DataFrame as a CSV file.
    predictions_with_ids.to_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_test_predicted.csv', index=False)

    # Print the RMSE and show the final DataFrame as a confirmation before saving it.
    print(f'Algorithm : RandomForestRegressor , RMSE in the validation set: {rmse}')
    print('DataFrame that will be saved as CSV:')
    print(predictions_with_ids.head())
    print(f'Total time taken: {int(minutes)} minutes and {int(seconds)} seconds')
    print('--RandomForestRegressor Algorithm done--')


# def execute_SVR():
#     print('--SVR Algorithm starting--')
#     start_time = time.time() #tracking start!
#     # Load training data and exclude the 'id' column.
#     X_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_train.csv').drop('id', axis=1)
#     y_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_train.csv')['score']

#     # Data normalization: it's important for SVR.
#     scaler_X = StandardScaler()
#     scaler_y = StandardScaler()

#     X_train_scaled = scaler_X.fit_transform(X_train)
#     y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

#     # Split the training data into a training set and a validation set.
#     X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train_scaled, test_size=0.2, random_state=42)

#     # Define a range of hyperparameters to search.
#     parameters = {
#         'C': [0.1, 1, 10, 100],  # Regularization parameter.
#         'gamma': ['scale', 'auto'],  # Kernel coefficient.
#         'epsilon': [0.01, 0.1, 0.2, 0.5, 1],  # Epsilon in the epsilon-SVR model.
#     }

#     # Instantiate the SVR model.
#     svr = SVR()

#     # Instantiate GridSearchCV.
#     grid_search = GridSearchCV(svr, parameters, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

#     # Execute the grid search.
#     grid_search.fit(X_train_split, y_train_split)

#     # Print the best parameters found.
#     print(f"Best parameters: {grid_search.best_params_}")

#     # Get the best model.
#     best_svr = grid_search.best_estimator_

#     # Load the test dataset and save the 'id'.
#     X_test = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_test.csv')
#     test_ids = X_test['id']
#     X_test_scaled = scaler_X.transform(X_test.drop('id', axis=1))

#     # Make predictions on the scaled test dataset with the best model.
#     test_predictions_scaled = best_svr.predict(X_test_scaled)

#     # Denormalize the predictions to get the final score.
#     test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).ravel()

#     # Create a DataFrame with the IDs and predictions.
#     predictions_with_ids = pd.DataFrame({
#         'id': test_ids,
#         'score': test_predictions
#     })

#     end_time = time.time()  # Finaliza el seguimiento del tiempo.
#     duration = end_time - start_time
#     minutes, seconds = divmod(duration, 60)

#     # Save the DataFrame as CSV.
#     predictions_with_ids.to_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_test_predicted_svr.csv', index=False)

#     # Calculate the RMSE in the validation set with the best model.
#     val_predictions_scaled = best_svr.predict(X_val_split)
#     val_predictions = scaler_y.inverse_transform(val_predictions_scaled.reshape(-1, 1)).ravel()
#     rmse = sqrt(mean_squared_error(y_val_split, val_predictions))
#     print(f'Algorithm: SVR,RMSE in the validation set: {rmse}')

#     # Print the final DataFrame as confirmation before saving it.
#     print('DataFrame that will be saved as CSV:')
#     print(predictions_with_ids.head())
#     print(f'Total time taken: {int(minutes)} minutes and {int(seconds)} seconds')
#     print('--SVR Algorithm done--')



def execute_ID3():
    print('--ID3 Algorithm starting--')
    start_time = time.time() #tracking start!
    # Load training data and exclude the 'id' column.
    X_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_train.csv').drop('id', axis=1)
    y_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_train.csv')['score']

    # Load the test dataset and save the 'id'.
    X_test = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_test.csv')
    test_ids = X_test['id']
    X_test = X_test.drop('id', axis=1)

    # Split the training data into a training set and a validation set.
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Instantiate the decision tree model for regression.
    model = DecisionTreeRegressor(random_state=42)

    # Train the model with the training data.
    model.fit(X_train_split, y_train_split)

    # Make predictions on the validation set to evaluate the model.
    val_predictions = model.predict(X_val_split)

    # Calculate the root mean squared error (RMSE) to assess the performance.
    rmse = sqrt(mean_squared_error(y_val_split, val_predictions))

    # If you are satisfied with the model's performance, train it with all the training data.
    model.fit(X_train, y_train)

    # Make predictions on the test dataset.
    test_predictions = model.predict(X_test)

    # Create a DataFrame with the IDs and predictions.
    predictions_with_ids = pd.DataFrame({
        'id': test_ids,
        'score': test_predictions
    })

    end_time = time.time()  # Finaliza el seguimiento del tiempo.
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    # Save the DataFrame as a CSV file.
    predictions_with_ids.to_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_test_predicted_id3.csv', index=False)

    # Print the RMSE and show the final DataFrame as confirmation before saving it.
    print(f'Algorithm : ID3,RMSE in the validation set: {rmse}')
    print('DataFrame that will be saved as CSV:')
    print(predictions_with_ids.head())
    print(f'Total time taken: {int(minutes)} minutes and {int(seconds)} seconds')
    print('--ID3 Algorithm done--')


def execute_XGBOOST():
    print('--XGBOOST Algorithm starting--')
    start_time = time.time() #tracking start!
    # Load training data and exclude the 'id' column.
    X_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_train.csv').drop('id', axis=1)
    y_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_train.csv')['score']

    # Split the training data into a training set and a validation set.
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Parameters to test
    param_distributions = {
        'n_estimators': randint(100, 1000),
        'colsample_bytree': uniform(0.5, 0.4),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.29),
        'alpha': uniform(0, 10)
    }

    # Instantiate the XGBoost model
    xgb_model = xgb.XGBRegressor(objective ='reg:squarederror')

    # Instantiate RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator = xgb_model,
        param_distributions = param_distributions,
        n_iter = 100,  # Number of parameter combinations to test
        scoring='neg_mean_squared_error',  # You can change this to another performance metric
        cv = 3,  # Number of cross-validation folds
        verbose=1,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )

    # Execute the random search
    random_search.fit(X_train_split, y_train_split)

    # Print the best parameters found
    print(f"Best parameters: {random_search.best_params_}")

    # Train the model with the best parameters on the entire training dataset
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Load the test dataset and save the 'id'.
    X_test = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_test.csv')
    test_ids = X_test['id']
    X_test = X_test.drop('id', axis=1)

    # Make predictions on the test dataset.
    test_predictions = best_model.predict(X_test)

    # Create a DataFrame with the IDs and predictions.
    predictions_with_ids = pd.DataFrame({
        'id': test_ids,
        'score': test_predictions
    })

    end_time = time.time()  # Finaliza el seguimiento del tiempo.
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    # Save the DataFrame as CSV.
    predictions_with_ids.to_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_test_predicted_xgboost.csv', index=False)

    # It's not necessary to calculate the RMSE here since RandomizedSearchCV already did it for us, but if you want...
    val_predictions = best_model.predict(X_val_split)
    rmse = sqrt(mean_squared_error(y_val_split, val_predictions))
    print(f"Algorithm : XGBOOST,Validation RMSE: {rmse}")

    # Show the final DataFrame as confirmation before saving it.
    print('DataFrame that will be saved as CSV:')
    print(predictions_with_ids.head())
    print(f'Total time taken: {int(minutes)} minutes and {int(seconds)} seconds')
    print('--XGBOOST Algorithm done--')


def execute_ANN():
    print('--ANN Algorithm starting--')
    start_time = time.time() #tracking start!
    # Load training data and exclude the 'id' column.
    X_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_train.csv').drop('id', axis=1)
    y_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_train.csv')['score']

    # Load the test dataset and save the 'id'.
    X_test = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_test.csv')
    test_ids = X_test['id']
    X_test = X_test.drop('id', axis=1)

    # Normalize the features since neural networks are sensitive to the scale of the inputs.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split the training data into a training set and a validation set.
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Build the neural network model.
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Single output for regression.

    # Compile the model - regression means we will use mean squared error as the loss function.
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model.
    model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), epochs=50, batch_size=32)

    # Make predictions on the validation set to evaluate the model.
    val_predictions = model.predict(X_val_split)

    # Calculate the root mean squared error (RMSE) to assess performance.
    rmse = sqrt(mean_squared_error(y_val_split, val_predictions))

    # Make predictions on the test dataset.
    test_predictions = model.predict(X_test_scaled)

    # Create a DataFrame with the IDs and predictions.
    predictions_with_ids = pd.DataFrame({
        'id': test_ids,
        'score': test_predictions.ravel()  # Make sure to flatten the array of predictions.
    })

    end_time = time.time()  # Finaliza el seguimiento del tiempo.
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    # Save the DataFrame as CSV.
    predictions_with_ids.to_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_test_predicted_ann.csv', index=False)

    # Print the RMSE and show the final DataFrame as confirmation before saving it.
    print(f'Algorithm: ANN,RMSE in the validation set: {rmse}')
    print('DataFrame that will be saved as CSV:')
    print(predictions_with_ids.head())
    print(f'Total time taken: {int(minutes)} minutes and {int(seconds)} seconds')
    print('--ANN Algorithm DONE--')




def execute_KNN():
    print('--KNN Algorithm starting--')
    start_time = time.time() #tracking start!
    # Load training data and exclude the 'id' column.
    X_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_train.csv').drop('id', axis=1)
    y_train = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_train.csv')['score']

    # Normalize the features, as KNN benefits from having all features on the same scale.
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    # It is not necessary to scale the target variables for KNN regression.

    # Split the training data into a training set and a validation set.
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Define a range of hyperparameters to search.
    parameters = {
        'n_neighbors': range(1, 30),  # Number of neighbors to consider.
        'weights': ['uniform', 'distance'],  # Weights of the points.
        'p': [1, 2]  # Type of distance to use: 1 is Manhattan, and 2 is Euclidean.
    }

    # Instantiate the KNN model.
    knn = KNeighborsRegressor()

    # Instantiate GridSearchCV.
    grid_search = GridSearchCV(knn, parameters, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

    # Execute the grid search.
    grid_search.fit(X_train_split, y_train_split)

    # Print the best parameters found.
    print(f"Best parameters: {grid_search.best_params_}")

    # Get the best model.
    best_knn = grid_search.best_estimator_

    # Load the test dataset and save the 'id'.
    X_test = pd.read_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_X_test.csv')
    test_ids = X_test['id']
    X_test_scaled = scaler_X.transform(X_test.drop('id', axis=1))

    # Make predictions on the scaled test dataset with the best model.
    test_predictions = best_knn.predict(X_test_scaled)

    # Create a DataFrame with the IDs and predictions.
    predictions_with_ids = pd.DataFrame({
        'id': test_ids,
        'score': test_predictions
    })

    end_time = time.time()  # Finaliza el seguimiento del tiempo.
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    # Save the DataFrame as CSV.
    predictions_with_ids.to_csv('/Users/appleuser/Desktop/pazmany/data_mining_practice/assignment/pc_y_test_predicted_knn.csv', index=False)

    # Calculate the RMSE in the validation set with the best model.
    val_predictions = best_knn.predict(X_val_split)
    rmse = sqrt(mean_squared_error(y_val_split, val_predictions))
    print(f'Algorithm:KNN,RMSE in the validation set: {rmse}')

    # Print the final DataFrame as confirmation before saving it.
    print('DataFrame that will be saved as CSV:')
    print(predictions_with_ids.head())
    print(f'Total time taken: {int(minutes)} minutes and {int(seconds)} seconds')
    print('--KNN Algorithm done--')





execute_RandomForestRegressor()
#execute_SVR()
execute_ID3()
execute_XGBOOST()
execute_ANN()
execute_KNN()









