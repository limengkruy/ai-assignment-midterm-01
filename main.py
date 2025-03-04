from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI()

# Disable GPU (Optional)
tf.config.set_visible_devices([], 'GPU')  # Disable default GPU setting

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is now enabled with Metal API")
    
# Load your pre-trained Logistic Regression model
try:
    logi = joblib.load("modeling/model/training/LogisticRegression.pkl")
    ann = tf.keras.models.load_model("modeling/model/training/ANN_10_Epochs.keras")
except Exception as e:
    raise RuntimeError("Failed to load the logistic model. Check the file path and try again.") from e

# Define the data model for incoming JSON requests
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def preprocess_data(data: CustomerData):
    # Convert the incoming JSON data to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Label encoding for binary/ordinal features
    label_map = {
        'gender': {'Female': 0, 'Male': 1},
        'SeniorCitizen': {0: 0, 1: 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    for col, mapping in label_map.items():
        input_df[col] = input_df[col].map(mapping)
    
    # One-Hot encode the nominal features (drop_first=True to avoid multicollinearity)
    input_df = pd.get_dummies(input_df, columns=[
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod'
    ], drop_first=True)
    
    # Use your fixed expected_columns list that matches your training output:
    expected_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # Reindex the DataFrame to ensure it has exactly the expected columns; fill missing ones with 0.
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    return input_df

@app.post("/predict/")
def predict(data: CustomerData):
    try:
        # Preprocess the incoming data
        processed_data = preprocess_data(data)
        # (Optional) Debug prints
        print("Processed data columns:", processed_data.columns.tolist())
        print("Processed data shape:", processed_data.shape)
        print("Processed data:", processed_data.values)
        # Make prediction
        _response = {}
        prediction = logi.predict(processed_data.values)
        print("Logistic prediction:", prediction)
        result = "Yes" if prediction[0] == 1 else "No"
        _response['prediction-logistic'] = result
        prediction = (ann.predict(processed_data.values) > 0.5).astype(int)
        print("ANN prediction:", prediction)
        result = "Yes" if prediction[0][0] == 1 else "No"
        _response['prediction-ann'] = result
        return _response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict-logi/")
def predict_logi(data: CustomerData):
    try:
        # Preprocess the incoming data
        processed_data = preprocess_data(data)
        # (Optional) Debug prints
        print("Processed data columns:", processed_data.columns.tolist())
        print("Processed data shape:", processed_data.shape)
        print("Processed data:", processed_data.values)
        # Make prediction
        prediction = logi.predict(processed_data.values)
        result = "Yes" if prediction[0] == 1 else "No"
        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
@app.post("/predict-ann/")
def predict_ann(data: CustomerData):
    try:
        # Preprocess the incoming data
        processed_data = preprocess_data(data)
        # (Optional) Debug prints
        print("Processed data columns:", processed_data.columns.tolist())
        print("Processed data shape:", processed_data.shape)
        print("Processed data:", processed_data.values)
        # Make prediction
        prediction = (ann.predict(processed_data.values) > 0.5).astype(int)
        result = "Yes" if prediction[0] == 1 else "No"
        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
