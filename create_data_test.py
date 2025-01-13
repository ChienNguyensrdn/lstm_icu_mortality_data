from data_process import DataProcess
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow as tf



def main():
    '''
    Main function to create data
    Cau hinh cac tham so de tao du lieu
    '''
    # Example data
    data = pd.DataFrame({
        'Time': ['00:01', '00:02', '00:04', '00:07', '00:11'],
        'Value': [10, 20, 30, 40, 50],
        'Parameter': ['HR', 'HR', 'HR', 'HR', 'HR']
    })
    processor = DataProcess(data)
    file_path = "DataInput/set-c/set-c/152871.txt"
    data_from_file = processor.read_file(file_path)
    print("Data from file:")
    print(data_from_file)
      # Test convert_time_to_step method
    step_data = processor.convert_time_to_step()
    print("Converted time to step:",len(step_data))
    print(step_data)
     # Test fill_missing_values method
    filled_data = processor.fill_missing_values()
    print("Filled missing values:", len(filled_data))
    print(filled_data)

    # Test XGBRegressor_predict_missing method
    predicted_data = processor.XGBRegressor_predict_missing()
    # predicted_data.to_csv('DataOutput/XGBRegressor_predict_missing.csv')

    print("Predicted missing values:",len(predicted_data))
    print(predicted_data)
    processor.combined_df = predicted_data

    dataset = processor.combined_df.copy()
    dataset = processor.combined_df[['Predict']]
    dataset = processor.scaler.fit_transform(np.array(dataset).reshape(-1, 1))
    test_dataset_df = processor.create_dataset(dataset)
    
    # Convert test_dataset_df to the required 3D shape for LSTM
    X_test = np.array(test_dataset_df['X'].tolist())
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # Assuming 1 feature
    
    LSTM = tf.keras.models.load_model("Models/lstm_hr.keras")
    test_predict = LSTM.predict(X_test)
    mse = mean_squared_error(test_dataset_df['y'], test_predict)
    print('MSE:', mse)
    test_predict = processor.scaler.inverse_transform(test_predict)
    print(test_predict)
    # Inverse transform the original dataset
    dataset_inverse = processor.scaler.inverse_transform(dataset)
    # np.savetxt('DataOutput/dataset_inverse.csv', dataset_inverse, delimiter=',')
# Convert X_test to a list of lists for DataFrame compatibility
    X_test = processor.scaler.inverse_transform(X_test.reshape(X_test.shape[0], X_test.shape[1]))
    X_test_list = [x.flatten().tolist() for x in X_test]
    y_test = processor.scaler.inverse_transform(test_dataset_df['y'].values.reshape(-1, 1))
    df =pd.DataFrame({'X': X_test_list, 'y': y_test.tolist(), 'Predict': test_predict.flatten()})
    df.to_csv('DataOutput/dataset_inverse.csv')
if __name__ == "__main__":
    main()