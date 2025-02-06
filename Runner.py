import errno
import os
import pickle
import logging
import time
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
    data = pd.DataFrame({
        'Time': ['00:01', '00:02', '00:04', '00:07', '00:11'],
        'Value': [10, 20, 30, 40, 50],
        'Parameter': ['HR', 'HR', 'HR', 'HR', 'HR']
    })
    processor = DataProcess(data)
    folder_path = '/home/chiennguyen/workspaces/Paper/DataInput/set-a/set-a'
    # processor.combined_df = processor.create_data_folder_path(folder_path)
    # def testfun(a, b, s):
    #     a = a + 1
    #     b = b / 2
    #     s = str.upper(s)
    #     return a, b, s


    # r = Runner(processor.create_data_folder_path, params={'folder_path': '/home/chiennguyen/workspaces/Paper/DataInput/set-a/set-a'}, cachePath='Results/', logfile='Results/test_log.csv')

    # processor.combined_df = r.run()


    rf = RunnerFactory(function=processor.create_data_folder_path, cachePath='cache/', logfile='cache/test_log.csv')

    configs = [
        {'params': {'folder_path': '/home/chiennguyen/workspaces/Paper/DataInput/set-a/set-a'},
         'name': 'processor.create_data_folder_path_seta'},
        {'params': {'folder_path': '/home/chiennguyen/workspaces/Paper/DataInput/set-c/set-c'},
         'name': 'processor.create_data_folder_path_setc'},
        {'params': {'folder_path': '/home/chiennguyen/workspaces/Paper/DataInput/set-b/set-b'},
         'name': 'processor.create_data_folder_path_setb'},
    ]

    for config in configs:
        print(rf.run(**config))
