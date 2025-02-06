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
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import logging
import pickle

class DataProcess():
    def __init__(self, data):
        self.data = data
        self.parameter = 'HR'
        self.interval = 10
        self.look_back = 3
        self.train_size = 0.7
        self.combined_df = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.epochs = 20
        self.batch_size = 32
        self.dropout = 0.2
        self.regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.early_stopping = EarlyStopping(
            monitor='val_loss',  # Giám sát validation loss
            patience=20,           # Số epoch không cải thiện liên tiếp trước khi dừng
            restore_best_weights=True  # Khôi phục trọng số tốt nhất đã lưu
        )
        self.scaler= MinMaxScaler(feature_range=(0, 1))

    def process(self):
        # do some process
        
        return self.data
    def read_file (self, file_path:str) -> pd.DataFrame:
        # read file theo path : Paper/DataInput/set-a/set-a/132539.txt
        #  trả về  dataframe có cac thuoc tinh sau :[time, value, parameter]
        self.data = pd.read_csv(file_path)
        return self.data
    def convert_time_to_step(self) -> pd.DataFrame:
        # chuyển thời gian về step
        # input : self.parameter
        # step = 0 tương ứng với thời gian đầu tiên của parameter
        # step = 1 tương ứng với thời gian thứ 2 của parameter - thời gian đầu tiên của parameter
        # step = 2 tương ứng với thời gian thứ 3 của parameter - thời gian thứ 2 của parameter
        # ... cho đến hết
        # output : dataframe có các thuộc tính sau : [Time, Step, Value]
        # chuyển đổi time sang số phút nguyên tắc split df['Time'] thành 2 phần hh:mm 
        # số phút = 60*hh + mm
        # merge voi du lieu co san theo parameter
        df = self.data[self.data['Parameter'] == self.parameter].copy()
        df['Minutes'] = df['Time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
        df['Step'] = df['Minutes'].diff().fillna(0).cumsum().astype(int)
        return df[['Time','Step', 'Value']]
    def fill_missing_values(self) -> pd.DataFrame:
        # min Step = 0
        # max Step = max(df['Step'])
        # tạo chuỗi step từ min đến max với interval = self.interval 
        # nếu step nào không có giá trị thì điền giá trị -1(đại diện cho NAN)
        # output : dataframe có các thuộc tính sau : [time,step, value]
        try:
            df = self.convert_time_to_step()
            min_step = 0
            max_step = df['Step'].max()
            all_steps = pd.DataFrame({'Step': np.arange(min_step, max_step + 1, self.interval)})
            df = pd.merge(all_steps, df, on='Step', how='left').fillna(-1)
            return df[['Time','Step', 'Value']]
        except Exception as e:
            return [] 
    def XGBRegressor_predict_missing(self) -> pd.DataFrame:
        # dự đoán giá trị thiếu (-1) bằng XGBRegressor
        # input : parameter
        # output : dataframe có các thuộc tính sau : [time,step, value]
        df = self.fill_missing_values()
        if len(df) == 0:
            return []
        train_df = df[df['Value'] != -1]
        test_df = df[df['Value'] == -1]

        X_train = train_df[['Step']]
        y_train = train_df['Value']
        X_test = test_df[['Step']]

        self.regressor =  XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.regressor.fit(X_train, y_train)
        # predictions = model.predict(X_test)
        Predicts =[]
        for index, row in df.iterrows():
            if row['Value'] == -1:
                Predicts.append(self.regressor.predict([row['Step']])[0])
            else:
                Predicts.append( row['Value'])
        df['Predict'] = Predicts

        # df.loc[df['Value'] == -1, 'Predict'] = model.predict(df[df['Value'] == -1]['Step'])
        # df.loc[df['Value'] != -1, 'Predict'] = df['Value'] 
        # print(df[['Time','Step', 'Value','Predict']])
        return df[['Time','Step', 'Value','Predict']]

    def create_data_folder_path(self, folder_path: str) -> pd.DataFrame:
        '''
        Đọc tất cả các file trong folder_path
        Input : folder_path
        output : dataframe có các thuộc tính sau : ['Id','Time','Value','Parameter','Predict']
        Id : tên file loai bỏ đuôi .txt
        Description : đọc từng file trong folder_path, sau đó thực hiện các bước sau :
        - convert_time_to_step
        - fill_missing_values
        - XGBRegressor_predict_missing
        - sau đó gộp các file lại thành 1 dataframe
        - trả về dataframe
        '''
        all_data = []
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file_name in tqdm(files, desc="Processing files", unit="file"):
            
            file_path = os.path.join(folder_path, file_name)
            self.read_file(file_path)
            parameter = self.data['Parameter'].iloc[0]  # Assuming all rows have the same parameter
            predicted_df = self.XGBRegressor_predict_missing( )
            if len(predicted_df) == 0:
                continue
            predicted_df['Time'] = self.data['Time']
            predicted_df['Parameter'] = self.parameter 
            predicted_df['Step'] = predicted_df['Step']
            predicted_df['Predict'] = predicted_df['Value']
            predicted_df['Id'] = file_name.replace('.txt', '')
            all_data.append(predicted_df)

        combined_df = pd.concat(all_data, ignore_index=True)
        self.combined_df = combined_df
        return combined_df[['Id', 'Time','Step', 'Value', 'Parameter', 'Predict']]
    def create_dataset(self,dataset ):
        X, y = [], []
        for i in range(len(dataset)-self.look_back-1):
            a = dataset[i:(i+self.look_back), 0]
            X.append(a)
            y.append(dataset[i + self.look_back, 0])
        return pd.DataFrame({'X': X, 'y': y})
    
    def Model_LSTM(self, regressor:str= None):
        '''
        Xây dựng mô hình LSTM để dự đoán du liệu tra ve tu ham create_dataset
        Goi y cac siêu tham số cần thiết
        Input : self.parameter
        Cac buoc thuc hien :
        - Tạo dataset voi ham create_data_folder_path thu duoc combined_df ->pd.DataFrame
        - Chia train va test theo self.train_size
            - train_size = int(len(combined_df) * self.train_size)
            - test_size = len(combined_df) - train_size
            - train, test = combined_df[0:train_size], combined_df[train_size:len(combined_df)]
        - goi ham create_dataset voi train, test, parameter look_back = self.look_back ->np.array
        - reshape X_train, X_test theo [samples, time_steps, features]
        - scaler = MinMaxScaler(feature_range=(0, 1)) dùng để chuẩn hóa dữ liệu
        - Create and fit the LSTM network
        Output : None
        '''
        # Create dataset
        # combined_df = self.create_data_folder_path('/path/to/folder')  # Replace with actual folder path
        
        # Split into train and test sets
        print(self.combined_df)
        dataset = self.combined_df[self.combined_df['Parameter'] == self.parameter].copy()
        dataset = self.combined_df[['Predict']]
        dataset = self.scaler.fit_transform(np.array(dataset).reshape(-1, 1))
        train_size = int(len(dataset) * self.train_size)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
        
        # Create datasets for train and test
        train_dataset = self.create_dataset(train)
        test_dataset = self.create_dataset(test)
        
        # Extract X and y
        self.X_train = np.array(train_dataset['X'].tolist())
        self.y_train = np.array(train_dataset['y'].tolist())
        self.X_test = np.array(test_dataset['X'].tolist())
        self.y_test = np.array(test_dataset['y'].tolist())
        
        # Debug prints
        print(f"X_train shape before reshaping: {self.X_train.shape}")
        print(f"X_test shape before reshaping: {self.X_test.shape}")
        
        # Check if X_train and X_test are not empty
        if self.X_train.size == 0 or self.X_test.size == 0:
            print("Error: X_train or X_test is empty.")
            return
        
        # Reshape input to be [samples, time steps, features]
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
    
       
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(self.look_back, 1)))
        model.add(LSTM(50))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        
        # Train the model
        model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_data=(self.X_test, self.y_test),
            callbacks=[self.early_stopping]
            )
        
        # Print model summary
        model.summary()
        self.model = model
        y_pred = model.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        mse = mean_squared_error(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        model_file = 'LSTM'+self.parameter+'_'+regressor+'.keras'
        model.save('Models/'+model_file)
        logging.info(f"LSTM Mean Squared Error {regressor} : {mse}")
        logging.info(f"LSTM Mean absolute error {regressor} : {mae}")
        return model

    def Model_ARIMA(self, regressor:str):
        '''
        Xây dựng mô hình ARIMA để dự đoán dữ liệu
        Input : self.parameter
        Cac buoc thuc hien :
        - Tạo dataset voi ham create_data_folder_path thu duoc combined_df ->pd.DataFrame
        - Chia train va test theo self.train_size
            - train_size = int(len(combined_df) * self.train_size)
            - test_size = len(combined_df) - train_size
            - train, test = combined_df[0:train_size], combined_df[train_size:len(combined_df)]
        - scaler = MinMaxScaler(feature_range=(0, 1)) dùng để chuẩn hóa dữ liệu
        - Create and fit the ARIMA model
        Output : None
        '''
        # Create dataset
        dataset = self.combined_df[self.combined_df['Parameter'] == self.parameter].copy()
        dataset = self.combined_df[['Predict']]
        dataset = self.scaler.fit_transform(np.array(dataset).reshape(-1, 1))
        train_size = int(len(dataset) * self.train_size)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
         # Extract X and y
        self.X_train = train
        self.y_train = train
        self.X_test = test
        self.y_test = test
        # Fit ARIMA model
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.forecast(steps=test_size)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        test = self.scaler.inverse_transform(test.reshape(-1, 1))
        
        # Calculate MSE
        mse = mean_squared_error(test, predictions)
        mae = mean_absolute_error(test, predictions)
        logging.info(f"ARIMA Mean Squared Error {regressor} : {mse}")
        logging.info(f"ARIMA Mean absolute error {regressor} : {mae}")
        model_file = 'ARIMA'+self.parameter+'_'+regressor+'.pkl'

        with open("Models/"+model_file, "wb") as file:
            pickle.dump("Models/"+model_file, file)
        return model_fit
def main():
    # Example data
    data = pd.DataFrame({
        'Time': ['00:01', '00:02', '00:04', '00:07', '00:11'],
        'Value': [10, 20, 30, 40, 50],
        'Parameter': ['HR', 'HR', 'HR', 'HR', 'HR']
    })

    # Initialize DataProcess with example data
    processor = DataProcess(data)

    # Test read_file method
    file_path = '/home/chiennguyen/workspaces/Paper/DataInput/set-a/set-a/132539.txt'
    data_from_file = processor.read_file(file_path)
    print("Data from file:")
    print(data_from_file)

    # Test convert_time_to_step method
    step_data = processor.convert_time_to_step()
    print("Converted time to step:",len(step_data))
    print(step_data)
    step_data.to_csv('DataOutput/1_datadataset.csv')
    # Test fill_missing_values method
    filled_data = processor.fill_missing_values()
    print("Filled missing values:", len(filled_data))
    print(filled_data)
    filled_data.to_csv('DataOutput/2_datadataset.csv')
    # Test XGBRegressor_predict_missing method
    predicted_data = processor.XGBRegressor_predict_missing()
    print("Predicted missing values:",len(predicted_data))
    print(predicted_data)
    predicted_data.to_csv('DataOutput/3_datadataset.csv')

    # Test create_data_folder_path method
    folder_path = '/home/chiennguyen/workspaces/Paper/DataInput/set-a/set-a'
    processor.combined_df = processor.create_data_folder_path(folder_path)
    print("Combined data from folder:")
    # print(combined_data)

    # Test Model_LSTM method
    model = processor.Model_LSTM()
    model.save('Models/lstm_hr.keras')
    y_pred = model.predict(processor.X_test)
    mse = mean_squared_error(processor.y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    # Inverse transform predictions and actual values for MSE calculation
    # y_pred = model.predict(processor.X_test)
    # y_pred = processor.scaler_y.inverse_transform(y_pred)
    # y_test = processor.scaler_y.inverse_transform(processor.y_test.reshape(-1, 1))
    
    # # Calculate MSE
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")
if __name__ == "__main__":
    main()


'''
Giai phap hoi quy nao tot nhat 
So sanh giai phap dung hoi quy voi khong hoi quy xem co tot hon khong

'''