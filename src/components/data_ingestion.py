## code related to reading the data like transformation , etc, try to divide the data into train and test etc and create validation data
## read data and spilt into traning and testing

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd ##work with data frame
from sklearn.model_selection import train_test_split
from dataclasses import dataclass ## clas variables
## where to save nraw data and other data , in a class i.e. dataingestion class config

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("enter thge data ingestion method or component")
        try:
            df=pd.read_csv('/home/hp/Downloads/MLOPs/notebook/data/stud.csv')
            logging.info('read the dataset as data frame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()