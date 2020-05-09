import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from feature_engineering import feature_engineering


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")
data_X, data_Y = feature_engineering(data, test= True)

# Load model:
model = xgb.XGBRegressor()
with open(f"models/{model.__class__.__name__}.pkl", 'rb') as f:
    model = pickle.load(f)

# Prediction:
pred = np.expm1(model.predict(data_X))

prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = pred


# Export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)

