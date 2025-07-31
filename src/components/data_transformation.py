## handle data encoding ,etc
import sys ## helps in handle system level exception
from dataclasses import dataclass ## helps in store config adtaa more clearly

import os
import numpy as np
import pandas as pd
# sklearn module is helping in preprocessing i.e  clean and converting the data
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging



from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    ## when ever class object is created then this fucntion will be called first
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


        ## responsible for converting caterogical features into numerical and filling missing values and scalling all heavy task
    def get_data_transformer_object(self):
        '''
        This functio is responsible for data tansformation
        '''

        try:
            ## telling which coloumns belong to which category
            numerical_coloumn=["writing_score","reading_score"]
            categorical_coloumn=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            ## creaitng pipeline and handles if theres any missing value in data
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), ## handeling the missing value
                    ("scaler",StandardScaler()) ## standard sclaling of values
                ]

            
            )


            cat_pipeline=Pipeline(
                ## handle missing values in categorical features
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()), ## converts text to numbers using one hot encoding
                    ("scaler",StandardScaler(with_mean=False)) 
                ]    
                ## converting categorical to numerical features and handeling that
            )

            logging.info(f"categorical cloumn:{categorical_coloumn}")

            logging.info(f"categorical cloumn:{numerical_coloumn}")


## combining above both pipeline into one
            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_coloumn),
                    ("cat_pipeline",cat_pipeline,categorical_coloumn)
                
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("0obtaining preprocesing object")
            
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_column=["writing_score","reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on traiing dataframe and etsting dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                        ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
                        ]

            logging.info(f"saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
             raise CustomException(e, sys)