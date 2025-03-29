import pandas as pd
import joblib
import numpy as np

model=joblib.load('LogReg.pkl')
le=joblib.load('label_encoders.pkl')
column_transformer=joblib.load('column_transformer.pkl')
y_encoder=joblib.load('y_encoder.pkl')

state=input("Enter your state: ").capitalize()
N=input("Enter nitrogen content in soil(mg/kg): ")
P=input("Enter phosphorus content in soil(mg/kg): ")
K=input("Enter potassium content in soil(mg/kg): ")
temp=float(input("Enter temperature in the region(celcius): "))
hum=float(input("Enter humidity in the region(%): "))
ph=float(input("Enter ph of the soil: "))
rain=float(input("Enter rainfall in the region in a month: "))
soil=input("Enter soil type: ").capitalize()
land=float(input("Enter your land size in acres: "))

if(__name__=="__main__"):
    new_data=pd.DataFrame({
        'N':N,
        'P':P,
        'K':K,
        'temperature':temp,
        'humidity':hum,
        'ph':ph,
        'rainfall':rain,
        'Soil Type':soil,
        'state':state,
        'land_size':land
        },index=[0])

def preprocess_new_data(X_new):
    # Apply label encoding
    for col, encoder in le.items():
        if col in X_new.columns:
            X_new[col] = encoder.transform(X_new[col])
    
    # Apply one-hot encoding
    if column_transformer:
        X_new = column_transformer.transform(X_new)
    
    return X_new

X_new_encoded = preprocess_new_data(new_data)

# Make predictions using your trained model
predictions = model.predict_proba(X_new_encoded)

top5_indices = np.argsort(predictions[0])[-5:][::-1]
top5_crops = y_encoder.inverse_transform(top5_indices)
top5_preds = predictions[0][top5_indices]

print("Top 5 Recommended Crops:")
for crop in zip(top5_crops):
    print(crop)



