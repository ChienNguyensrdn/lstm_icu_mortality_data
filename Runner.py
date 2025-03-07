import errno
import os
import pickle
import logging
import time
# from data_process import DataProcess
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

from lstm_time_series import LstmTimeSeries
from pipline_data_process import PiplineDataProcess
#from utils.telegramNotification import TeleNotifier
'''''''''''
# run a function with a set of params and return results as a list
# support caching and timing
'''''''''''


class Runner:
    task_id = 0

    def __init__(self, function, params, name=None, cachePath=None, logfile='DataOutput/runner_log.csv'):
        Runner.task_id = Runner.task_id + 1
        self.function = function
        self.params = params
        self.cachePath = cachePath

        if name is None:
            self.name = 'Task {}'.format(Runner.task_id)
        else:
            self.name = name
        if cachePath is not None:
            if not os.path.exists(cachePath):
                try:
                    os.makedirs(cachePath)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            self.cacheFile = os.path.join(cachePath, 'runner_' + self.name + '.runner_cache')
        else:
            self.cacheFile = None
        # init logger
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s, %(message)s')

    def run(self, ignoreCache=False):
        result = None
        if not ignoreCache and self.cacheFile is not None:
            try:
                with open(self.cacheFile, 'rb') as f:
                    result = pickle.load(f)
                    logging.info('{}, CACHE OK, {}'.format(self.name, self.cacheFile))
            except:
                logging.info('{}, CACHE FAILED, {}'.format(self.name, self.cacheFile))
                result = None
        if result is None:
            # run the task
            start_time = time.time()
            logging.info('LOG, started')
            result = self.function(**self.params)
            logging.info(self.name + ' , RUNTIME, %s' % (time.time() - start_time))
            # store value
            if self.cacheFile is not None:
                with open(self.cacheFile, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                    logging.info('{}, CACHE STORED, {}'.format(self.name, self.cacheFile))
        return result

    def clear_cache(self):
        if self.cacheFile is not None and os.path.exists(self.cacheFile):
            os.remove(self.cacheFile)


class RunnerFactory:

    def __init__(self, function, cachePath=None, logfile='DataOutput/runner_log.csv', ignoreCache=False):
        self.function = function
        self.cachePath = cachePath
        self.logfile = logfile
        self.ignoreCache = ignoreCache

    def run(self, params, name):
        runner = Runner(function=self.function, params=params,
                        name=name,
                        cachePath=self.cachePath,
                        logfile=self.logfile)
        return runner.run(ignoreCache=self.ignoreCache)


if __name__ == '__main__':
     # Example data
    
    # folder_path = '/home/chiennguyen/workspaces/Paper/DataInput/set-a/set-a'
    # processor.combined_df = processor.create_data_folder_path(folder_path)
    def run_pipline(folder_path):
        pipline = PiplineDataProcess()
        pipline.FolderPath = folder_path
        data_train_X, data_train_y =[],[]
        files = [f for f in os.listdir(pipline.FolderPath) if f.endswith('.txt')]
        Counter = 0
        Error = 0
        for file_name in tqdm(files, desc="Processing files", unit="file"):
            try:
                Counter +=1
                file_path = os.path.join(pipline.FolderPath, file_name)
                df = pipline.read_file(file_path)
                df = pipline.convert_time_to_step(df, islog=False)
                df = pipline.fill_missing_data_default(df, islog=False)
                pipline.Regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                df = pipline.recovering_missing_data_regression(df, islog=False)
                pipline.create_data_train_test(df,False)
                
            except:
                Error +=1
                continue
        # pd.DataFrame({'X': X, 'y': y,'miss' :miss}).to_csv('DataOutput/convert_time_step/create_data_train_test.csv')
        print('Done')
        print('Error : ', Error)
        print('Total : ', Counter)
        print('Success : ', Counter - Error)
        # return pd.DataFrame({'X': pipline.X, 'y': pipline.y,'miss' :pipline.miss})
        return pipline.X, pipline.y, pipline.miss
    def run_lstm(X,y,miss):
        lstm = LstmTimeSeries(n_features=1, n_steps=10, n_units=50, n_epochs=100, n_batch_size=32)
        X = np.array(X)
        y = np.array(y)
        miss = np.array(miss)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        rain_size = int(len(X) * 0.7)
        X_train, X_test, y_train, y_test = X[:rain_size], X[rain_size:], y[:rain_size], y[rain_size:]

        miss_train_y, miss_test_y = miss[:rain_size], miss[rain_size:]

        model = lstm.build_model()
        model = lstm.fit_model(model, X_train, y_train)
        y_pred = lstm.predict(model, X_test)
        mse = lstm.evaluate(y_test, y_pred, y_miss=miss_test_y)
        print('Mean Squared Error: %.2f' % mse)
        return mse
        

        

    rf = RunnerFactory(function=run_pipline, cachePath='cache/', logfile='cache/test_log.csv')
    rlstm = RunnerFactory(function=run_lstm, cachePath='cache/', logfile='cache/test_log.csv')
    configs = [
        {'params': {'folder_path': 'DataInput/set-a/set-a'},
         'name': 'run_pipline.create_data_folder_path_seta'},
        {'params': {'folder_path': 'DataInput/set-c/set-c'},
         'name': 'run_pipline.create_data_folder_path_setc'},
        {'params': {'folder_path': 'DataInput/set-b/set-b'},
         'name': 'run_pipline.create_data_folder_path_setb'},
    ]
    i = 1
    for config in configs:
        X, y, miss = rf.run(**config)
        # df.to_csv(f'DataOutput/convert_time_step/{i}___create_data_train_test.csv')
        i+=1
        
        print(rf.run(**config))
        print(rlstm.run(**{'params':{'X':X,'y':y, 'miss':miss}, 'name':f'{i}___run_lstm.run'}))
        
        # rlstm.run(df=df)
        exit()
