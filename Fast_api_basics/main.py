import numpy as np
import pandas as pd
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import csv

# Step 1: Train a simple ML model (Linear Regression on Diabetes dataset)
MODEL_FILE = "diabetes_model1.pkl"
CSV_FILE = "diabetes_data1.csv"

if not os.path.exists(MODEL_FILE):
    # Load Diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Split data into train/test sets
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    
    # Feature scaling   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled,y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    # comparision stats
    comparision_df = pd.DataFrame(
        {"Actual":y_test, "Predicted":y_pred}
    )
    comparision_df.to_csv(CSV_FILE, index = False)
    
    # metric calculations
    mse = mean_squared_error(y_test,y_pred)
    r2= r2_score(y_test,y_pred)
    print(f"Model trained. MSE: {mse}, R2 Score: {r2}")

    # saved the trained model
    joblib.dump((model, scaler), MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
else:
    print(f"Model already exists at {MODEL_FILE}")

# Step 2: Create FastAPI app
app = FastAPI(title="Diabetes Prediction API")

# Step 3: Load model at startup
model, scaler = joblib.load(MODEL_FILE)

# Step 4: Create root route /
@app.get("/")
async def root():
    return {"Status": "API is Running oho"}


# Step 5: Define Pydantic model for input
class DiabetesInput(BaseModel):
    age:float
    sex :float
    bmi :float
    bp:float
    s1:float
    s2:float
    s3:float
    s4:float
    s5:float
    s6:float

# Step 6 & 7: Create /predict endpoint
@app.post("/predict")
async  def predict(input_data:DiabetesInput):
    try:
        # convert input to numpy array
        features = np.array([ list(input_data.model_dump().values())])
        features_scaled = scaler.transform(features)
        # make prediction
        prediction = model.predict(features_scaled)[0]
        return {"Prediction" : prediction}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail =f"Error due to {e}")




