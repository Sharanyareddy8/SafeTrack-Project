import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("SafeTrack: Crime Risk Detection and Safe Route Suggestion")

# Load dataset
data = pd.read_excel("hyderabad_crime_dataset.xlsx")

# Clean column names
data.columns = data.columns.str.strip()

# Identify important columns safely
area_col = data.columns[0]
crime_col = data.columns[1]
time_col = data.columns[2]
target_col = data.columns[-1]

# Encode dataset
data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.iloc[:, :-1]
y = data_encoded.iloc[:, -1]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# UI
areas = data[area_col].unique()
time_slots = data[time_col].unique()

source = st.selectbox("Select Source Area", areas)
destination = st.selectbox("Select Destination Area", areas)
time = st.selectbox("Select Time Slot", time_slots)

if st.button("Check Safety"):

    avg_crime = data[data[area_col] == destination][crime_col].mean()

    input_data = pd.DataFrame({
        area_col: [destination],
        crime_col: [avg_crime],
        time_col: [time]
    })

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    st.subheader("Predicted Risk Level:")
    st.write(prediction)

    if prediction == "High":
        st.error("⚠ High risk area detected. Suggesting alternate safe route.")
    else:
        st.success("✔ Route considered relatively safe.")
    
