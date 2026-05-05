import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
import joblib
import os 

# File paths 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data","churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "model.pkl")

# Data pre-processing

def load_data():
    df =pd.read_csv(DATA_PATH)
    print(f"DATA loaded: {df.shape}")
    return df

def preprocess_data(df):
    df = df.copy()
    if "customerID" in df.columns:
        df =df.drop(columns =["customerID"])
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors= "coerce")
    
    df=df.fillna(0)
    
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    df=pd.get_dummies(df, drop_first= True)

    return df

#   Prediction Model 

def train_model(df):
    X =df.drop(columns= ["Churn"])
    y= df["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42
                                                     )
    
    model= RandomForestClassifier(n_estimators=100,random_state=42)

    model.fit(X_train,y_train)

    y_pred= model.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test,y_pred):.4f}")
    print(f"Recall: {recall_score(y_test,y_pred):.4f}")
    print(f"Precision:{precision_score(y_test,y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test,model.predict_proba(X_test)[:,1]):.4f}")

    y_proba= model.predict_proba(X_test)[:,1]
    y_pred_tuned= (y_proba >= 0.3).astype(int)

    print(f"\n After threshold tuning (0.3):")

    print(f"Recall: {recall_score(y_test,y_pred_tuned):.4f}")
    
    return model

def save_model(model):
    joblib.dump(model,MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    df =load_data()
    df=preprocess_data(df)
    model =train_model(df)
    save_model(model)
    print("Training complete!")
















