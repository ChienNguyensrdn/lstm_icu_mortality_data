
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor


class PiplineDataProcess:
    def __init__(self):
        self.FolderPath = "DataInput/set-a/set-a"
        self.Code = "HR"
        self.Interval = 15
        self.Regressor = None
        self.look_back =10
        self.X = []
        self.y = []
        self.miss = []

    def read_file(self, file_path:str) -> pd.DataFrame:
        '''
        Read data from file
        '''
        return pd.read_csv(file_path)
    
    def convert_time_to_step(self, df: pd.DataFrame, islog:False) -> pd.DataFrame:
        '''chuyển thời gian về step
        input : self.parameter
        step = 0 tương ứng với thời gian đầu tiên của parameter
        step = 1 tương ứng với thời gian thứ 2 của parameter - thời gian đầu tiên của parameter
        step = 2 tương ứng với thời gian thứ 3 của parameter - thời gian thứ 2 của parameter
        ... cho đến hết
        output : dataframe có các thuộc tính sau : [Time, Step, Value]
        chuyển đổi time sang số phút nguyên tắc split df['Time'] thành 2 phần hh:mm 
        số phút = 60*hh + mm
        merge voi du lieu co san theo parameter'''
        dataframe = df[df['Parameter'] == self.Code].copy()
        dataframe['Minutes'] = dataframe['Time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
        dataframe['Step'] = dataframe['Minutes'].diff().fillna(0).cumsum().astype(int)
        if islog:
            '''ghi log'''
            dataframe.to_csv('DataOutput/convert_time_step/convert_time_to_step.csv')
        return dataframe[['Time','Step', 'Value']]
    def fill_missing_data_default (self, df: pd.DataFrame, islog:False) -> pd.DataFrame:
        min_step = 0
        max_step = df['Step'].max()
        all_steps = pd.DataFrame({'Step': np.arange(min_step, max_step + 1, self.Interval)})
        df = pd.merge(all_steps, df, on='Step', how='left').fillna(-1)
        if islog:
            '''ghi log'''
            df.to_csv('DataOutput/convert_time_step/fill_missing_data_default.csv')
        return df[['Time','Step', 'Value']]
       
        
    def recovering_missing_data_regression (self, df: pd.DataFrame, islog:False) -> pd.DataFrame:
        '''phục hồi dữ liệu bị thiếu bằng cách sử dụng hồi quy tuyến tính
        input : dataframe sau khi đã fill_missing_data_default
        output : dataframe sau khi phục hồi dữ liệu bị thiếu'''
        # dự đoán giá trị thiếu (-1) bằng XGBRegressor
        # input : parameter
        # output : dataframe có các thuộc tính sau : [time,step, value]
        if len(df) == 0:
            return []
        train_df = df[df['Value'] != -1]
        test_df = df[df['Value'] == -1]

        X_train = train_df[['Step']]
        y_train = train_df['Value']
        X_test = test_df[['Step']]
        self.Regressor.fit(X_train, y_train)
        Predicts =[]
        for index, row in df.iterrows():
            if row['Value'] == -1:
                Predicts.append(self.Regressor.predict([row['Step']])[0])
            else:
                Predicts.append( row['Value'])
        df['Predict'] = Predicts
        if islog:
            '''ghi log'''
            df.to_csv('DataOutput/convert_time_step/recovering_missing_data_regression.csv')
        return df[['Time','Step', 'Value','Predict']]


    def create_data_train_test(self, df : pd.DataFrame, isLog = False) :
        '''tạo dữ liệu train và test
        input : dataframe sau khi đã phục hồi dữ liệu bị thiếu
        output : dataframe train và test'''
        # X, y,miss = [], [], []
        for i in range(len(df) - self.look_back - 1):
            a = df.iloc[i:(i + self.look_back), df.columns.get_loc('Predict')].values
            self.X.append(a)
            self.y.append(df.iloc[i + self.look_back, df.columns.get_loc('Predict')])
            self.miss.append(df.iloc[i + self.look_back, df.columns.get_loc('Value')])

        if isLog:
            '''ghi log'''
            pd.DataFrame({'X': self.X, 'y': self.y,'miss' :self.miss}).to_csv('DataOutput/convert_time_step/create_data_train_test_train.csv')
    
# def run_pipline(path_folder = "DataInput/set-a/set-a"):
#     pipline = PiplineDataProcess()
#     pipline.FolderPath = path_folder
#     data_train_X, data_train_y =[],[]
#     files = [f for f in os.listdir(pipline.FolderPath) if f.endswith('.txt')]
#     X, y,miss = [], [], []
#     Counter = 0
#     Error = 0
#     for file_name in tqdm(files, desc="Processing files", unit="file"):
#         try:
#             Counter +=1
#             file_path = os.path.join(pipline.FolderPath, file_name)
#             df = pipline.read_file(file_path)
#             df = pipline.convert_time_to_step(df, islog=True)
#             df = pipline.fill_missing_data_default(df, islog=True)
#             pipline.Regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#             df = pipline.recovering_missing_data_regression(df, islog=True)
#             i1,i2,i3 = pipline.create_data_train_test(df,True)
#             X.append(i1)
#             y.append(i2)
#             miss.append(i3)
#         except:
#             Error +=1
#             continue
#     pd.DataFrame({'X': X, 'y': y,'miss' :miss}).to_csv('DataOutput/convert_time_step/create_data_train_test.csv')
#     print('Done')
#     print('Error : ', Error)
#     print('Total : ', Counter)
#     print('Success : ', Counter - Error)

# if __name__ == '__main__':
#     run_pipline()