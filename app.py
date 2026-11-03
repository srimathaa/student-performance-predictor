

import streamlit as st
import pandas as pd
import joblib

# Load saved model information
saved_data = joblib.load("best_model.pkl")

model = saved_data["model"]
features = saved_data["features"]
model_name = saved_data["model_name"]

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's final mathematics grade "
    "using academic and personal factors."
)

st.info(f"Selected Machine Learning Model: {model_name}")

# Student inputs
age = st.slider("Age", 15, 22, 17)

studytime = st.selectbox(
    "Weekly Study Time",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Less than 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "More than 10 hours"
    }[x]
)

failures = st.selectbox(
    "Number of Previous Failures",
    [0, 1, 2, 3]
)

absences = st.slider(
    "Number of School Absences",
    0, 75, 4
)

medu = st.selectbox("Mother's Education Level", [0, 1, 2, 3, 4])
fedu = st.selectbox("Father's Education Level", [0, 1, 2, 3, 4])

traveltime = st.selectbox(
    "Travel Time to School",
    [1, 2, 3, 4]
)

health = st.slider(
    "Current Health (1 = Very Bad, 5 = Very Good)",
    1, 5, 3
)

# Prediction
if st.button("Predict Final Grade"):

    input_data = pd.DataFrame(
        [[
            age,
            studytime,
            failures,
            absences,
            medu,
            fedu,
            traveltime,
            health
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(20, prediction))

    st.success("Prediction completed!")

    st.metric(
        "Predicted Final Grade",
        f"{prediction:.2f} / 20"
    )
