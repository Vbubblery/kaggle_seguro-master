from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
class Classifier(BaseEstimator):
    def __init__(self):
        self.model = XGBClassifier(    
                        n_estimators=400,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                     )
    def fit(self, X, y):
        self.model = self.model.fit(X,y)
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)