# importing the libraries
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging

# logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def evel_metrices(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae,r2



if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    #reading the csv file

    
    data = pd.read_csv("USA_Housing.csv")


    #splt the data into training and test split
    data.drop("Address", axis=1, inplace=True)
    train, test = train_test_split(data)
    # the prediction column is "quality" which is a scaler 
    train_x =train.drop(["Price"], axis = 1)
    test_x = test.drop(["Price"], axis = 1)
    train_y = train["Price"]
    test_y=test["Price"]

    alpha = float(sys.argv[1]) if len(sys.argv)>1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv)>2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state = 42)
        lr.fit(train_x, train_y)

        predictino_quantity = lr.predict(test_x)

        (rmse, mae,r2) = evel_metrices(test_y, predictino_quantity)


        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        prediction = lr.predict(test_x)
        signature = infer_signature(train_x, prediction)


        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        #model registery does not work with file store
        if tracking_url_type_store != "file":
            #tracking in uri
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticcnetWithModel", signature=signature
            )
        else:
            #tracking in local
            mlflow.sklearn.log_model(lr, "model", signature=signature)