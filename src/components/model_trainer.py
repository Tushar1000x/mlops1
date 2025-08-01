## training the data model, solving data regression ,  etc changing values, model pickel file to cloud
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,Y_train,X_test,Y_test=(train_array[:,:-1],## last coloumn and store it in x
                                           train_array[:,-1],
                                           test_array[:,:-1],
                                           test_array[:,-1]
                                           )
            model={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGB Regresor":XGBRegressor(),
                "K-Neighbours Regression":KNeighborsRegressor(),
                "Cat Boost Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
            }
            Params={
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGB Regresor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]              
                },
                "K-Neighbours Regression":{
                    'n_neighbors':[5,7,9,11],
                },
                "Cat Boost Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[.1,.01,.05,.001],
                    'iterations':[30,50,100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]              
                },
                
            }

            model_report:dict=evaluate_models(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test,models=model,params=Params)
            ## get best model score from dictionry
            best_model_score=max(sorted(model_report.values()))

            ## get best model name form dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=model[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model found")
            
            logging.info(f"best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            ## evaluating model by using r^2 test
            predicted=best_model.predict(X_test)

            r2_square=r2_score(Y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)