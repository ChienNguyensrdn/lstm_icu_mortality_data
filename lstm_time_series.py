import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LstmTimeSeries:
    def __init__(self, n_features, n_steps, n_units, n_epochs, n_batch_size):
        self.n_features = n_features
        self.n_steps = n_steps
        self.n_units = n_units
        self.n_epochs = n_epochs
        self.batch_size = n_batch_size
        self.dropout = 0.2
        self.early_stopping = EarlyStopping(
            monitor='val_loss',  # Giám sát validation loss
            patience=20,           # Số epoch không cải thiện liên tiếp trước khi dừng
            restore_best_weights=True  # Khôi phục trọng số tốt nhất đã lưu
        )

    def build_model(self):
        # model = Sequential()
        # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(self.look_back, 1)))
        # model.add(LSTM(50))
        # model.add(Dropout(self.dropout))
        # model.add(Dense(1))
        # model.compile(loss='mse', optimizer='adam')

        model = Sequential()
        model.add(LSTM(units=self.n_units, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(units=self.n_features))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit_model(self, model, X, y):
        X = np.array(X)
        y = np.array(y)
        model.fit(
            X, y, 
            epochs=self.n_epochs, 
            verbose=0,
            batch_size=self.batch_size,
            callbacks=[self.early_stopping]
        )
        return model

    def predict(self, model, X):
        return model.predict(X)

    def forecast(self, model, X, n_forecast):
        y_pred = []
        for i in range(n_forecast):
            y_pred.append(model.predict(X)[0])
            X = np.append(X, y_pred[-1].reshape(1, 1, self.n_features), axis=1)
            X = X[:, 1:, :]
        return np.array(y_pred)

    def evaluate(self, y_true, y_pred, y_miss):
        '''
        Đánh giá mô hình bằng cách so sánh giá trị dự đoán và giá trị thực tế
        :param y_true: Giá trị thực tế
        :param y_pred: Giá trị dự đoán
        :param y_miss: Giá trị bị thiếu
        :return: Mean Squared Error
        Neu y_miss = -1 thi khong tinh vao MSE
        '''
        # Filter out the missing values
        mask = y_miss != -1
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        return mean_squared_error(y_true_filtered, y_pred_filtered)