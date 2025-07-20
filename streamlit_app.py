import streamlit as st
import pandas as pd
import joblib
import requests
model = joblib.load("model/grade_predictor.pkl")
scaler = joblib.load("model/scaler.pkl")
grade_map = {5: 'A', 4: 'B', 3: 'C', 2: 'D', 1: 'E', 0: 'F'}

st.title("Student Grade Predictor")
def generate_feedback(input_data, predicted_grade):
    prompt = f"""
You are an academic advisor. A student has received grade {predicted_grade} in a university course.

Their profile is:
- Attendance: {input_data['Attendance (%)'].values[0]}%
- Midterm Score: {input_data['Midterm_Score'].values[0]}
- Assignments Average: {input_data['Assignments_Avg'].values[0]}
- Quizzes Average: {input_data['Quizzes_Avg'].values[0]}
- Participation Score: {input_data['Participation_Score'].values[0]}
- Projects Score: {input_data['Projects_Score'].values[0]}
- Study Hours per Week: {input_data['Study_Hours_per_Week'].values[0]}
- Stress Level (1–10): {input_data['Stress_Level (1-10)'].values[0]}
- Sleep Hours per Night: {input_data['Sleep_Hours_per_Night'].values[0]}
- Gender: {"Male" if input_data["Gender"].values[0]==1 else "Female"}
- Extracurricular Activities: {"Yes" if input_data["Extracurricular_Activities"].values[0]==1 else "No"}

Write a short, **personalized and encouraging feedback message**  acknowledging their current grade and suggesting **one practical tip** to improve or maintain performance in the future.
Avoid repeating the values directly and instead interpret them.
"""

    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        return result[0]["generated_text"].strip()
    except Exception as e:
        return "Feedback generation failed."

with st.form("grade_form"):
    gender = st.radio("Gender", ['Male', 'Female'])
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0)
    midterm = st.slider("Midterm Score", 0.0, 100.0, 75.0)
    assignments = st.slider("Assignments Average", 0.0, 100.0, 70.0)
    quizzes = st.slider("Quizzes Average", 0.0, 100.0, 65.0)
    participation = st.slider("Participation Score", 0.0, 100.0, 60.0)
    projects = st.slider("Projects Score", 0.0, 100.0, 80.0)
    study_hours = st.slider("Study Hours per Week", 0.0, 40.0, 10.0)
    extra = st.radio("Extracurricular Activities", ['Yes', 'No'])
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
    submitted = st.form_submit_button("Predict Grade")
if submitted:
    gender_val = 1 if gender == "Male" else 0
    extra_val = 1 if extra == "Yes" else 0
    input_data = pd.DataFrame([{
        "Gender": gender_val,
        "Attendance (%)": attendance,
        "Midterm_Score": midterm,
        "Assignments_Avg": assignments,
        "Quizzes_Avg": quizzes,
        "Participation_Score": participation,
        "Projects_Score": projects,
        "Study_Hours_per_Week": study_hours,
        "Extracurricular_Activities": extra_val,
        "Stress_Level (1-10)": stress,
        "Sleep_Hours_per_Night": sleep
    }])
    scale_cols = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Study_Hours_per_Week",
    "Stress_Level (1-10)",
    "Sleep_Hours_per_Night"]
    scaled_input = input_data.copy()
    HF_API_TOKEN = st.secrets["HF_TOKEN"]
    scaled_input[scale_cols] = scaler.transform(input_data[scale_cols])
    final_input = scaled_input[scale_cols]
    final_input.loc[:, "Gender"] = input_data["Gender"].values
    final_input.loc[:, "Extracurricular_Activities"] = input_data["Extracurricular_Activities"].values
    prediction = model.predict(final_input)[0]
    grade = grade_map[prediction]
    feedback = generate_feedback(input_data, grade)
    st.success(f"Predicted Grade: **{grade}**")
    st.info(f"Feedback:\n\n{feedback}")
