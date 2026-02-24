import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_excel("hyderabad_crime_dataset.xlsx")
data.columns = data.columns.str.strip()

# Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.iloc[:, :-1]
y = data_encoded.iloc[:, -1]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("SafeTrack: Crime Risk Detection and Safe Route Suggestion")

areas = data["Area"].unique()
time_slots = data["Time_Slot"].unique()

source = st.selectbox("Select Source Area", areas)
destination = st.selectbox("Select Destination Area", areas)
time = st.selectbox("Select Time Slot", time_slots)

if st.button("Check Safety"):

    input_data = pd.DataFrame({
        "Area": [destination],
        "Time_Slot": [time],
        "Crime_Count": [data[data["Area"] == destination]["Crime_Count"].mean()]
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
