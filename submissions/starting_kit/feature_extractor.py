import numpy as np
class FeatureExtractor():
    def __init__(self):
        pass
    def fit(self, X_df,y):
        pass
    def transform(self, X_df):
        ps_cal = X_df.columns[X_df.columns.str.startswith('ps_calc')]
        X_df = X_df.drop(ps_cal,axis =1)
        X_df = X_df.fillna(999)
        for c in X_df.select_dtypes(include=['float64']).columns:
            X_df[c]=X_df[c].astype(np.float32)
        for c in X_df.select_dtypes(include=['int64']).columns[2:]:
            X_df[c]=X_df[c].astype(np.int8)
        return X_df
