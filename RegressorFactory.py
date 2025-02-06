from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
class RegressorFactory:
    def __init__(self, regressor_type):
        self.regressor_type = regressor_type

    def create_regressor(self, custom_params = {}) :
        '''
        Factory method to create regressor
        Input params: custom_params: dict, custom parameters for regressor
        Output: regressor model
        Remark: custom_params is a dictionary, for example: {'n_estimators': 100, 'max_depth': 5}
                custom_params = None if using default parameters of regressor
        '''
        if self.regressor_type == 'RandomForest':
            return (RandomForestRegressor( **custom_params))
        elif self.regressor_type == 'Linear':
            return LinearRegression( **custom_params)
        elif self.regressor_type == 'Ridge':
            return Ridge( **custom_params) 
        elif self.regressor_type == 'Lasso':
            return Lasso( **custom_params) 
        elif self.regressor_type == 'ElasticNet':
            return ElasticNet(  **custom_params)
        elif self.regressor_type == 'XGB':
            return XGBRegressor( **custom_params)
        else:
            raise ValueError('Invalid regressor type')