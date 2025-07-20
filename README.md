## Student Grade Predictor with LLM Feedback

This is a Streamlit-based web app that predicts a student's final grade based on academic and lifestyle inputs and provides personalized feedback using an LLM (Large Language Model).

## Features

- Predicts grades (A–F) using a pre-trained ML model 
- Takes into account:
  - Attendance
  - Midterm score
  - Assignments, quizzes, projects
  - Participation
  - Study hours, stress, sleep, gender, extracurriculars
- Uses LLM to generate personalized, encouraging feedback based on input
- Clean and interactive UI with Streamlit


##  Tech Stack

- Python
- Streamlit
- scikit-learn / XGBoost
- Hugging Face Inference API (LLM)
- Pandas, Joblib

##  Installation

```bash
git clone https://github.com/your-username/student-grade-predictor.git](https://github.com/VaamanBansal/vinnovateit-tech-task-Vaaman_Bansal
cd student-grade-predictor
pip install -r requirements.txt

## Run the app
python -m streamlit run streamlit_app.py

##Secrets
To use Hugging Face or OpenAI LLM feedback, create a file at:
.streamlit/secrets.toml
With:
HF_TOKEN = "your_huggingface_token"

##Example Input
Gender: Male

Attendance: 85%

Midterm: 75

Sleep: 7 hrs

Study Hours: 10/week

Stress Level: 5/10

##Output
Predicted Grade: C

LLM Feedback: "You're doing well, but increasing your quiz preparation slightly might boost your results. Keep it up!"
