# 🧠 Personality Deduction App

A Machine Learning web application that predicts whether a person is **Extrovert, Introvert, or Ambivert** based on multiple personality traits.

## 🚀 Project Overview

This project uses a **Logistic Regression model** trained on a synthetic personality dataset.

Users can input their personality trait scores through an interactive **Streamlit UI**, and the model predicts their personality type.

### Prediction Classes

* Extrovert
* Introvert
* Ambivert

---

## 📊 Features Used

The model uses multiple personality-related features such as:

* Social Energy
* Alone Time Preference
* Talkativeness
* Deep Reflection
* Group Comfort
* Party Liking
* Listening Skill
* Empathy
* Leadership
* Risk Taking
* Curiosity
* Planning
* Spontaneity
* Travel Desire
* Gadget Usage
* Decision Speed
* and more...

Dropped features:

* creativity
* emotional_stability
* stress_handling

Target Column:

* personality_type

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Pickle

---

## Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Selection
4. Data Scaling using StandardScaler
5. Logistic Regression Model Training
6. Model Evaluation
7. Streamlit Deployment

---

## Model Performance

✅ Logistic Regression Accuracy: **99.75%**

> Note: Since the dataset is synthetic, very high accuracy may indicate highly separable patterns.

---

## Project Structure

```bash
personality-deduction-app/
│
├── app.py
├── personality_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/personality-deduction-app.git
cd personality-deduction-app
pip install -r requirements.txt
streamlit run app.py
```

---

## Live Demo

Add your deployed Streamlit link here after deployment.

Example:
[https://your-app-name.streamlit.app/](https://personality-deduction-meizhmfpxrk94kh4vvbrcv.streamlit.app/)

---

## Screenshots

Add your Streamlit UI screenshots here.

---

## Future Improvements

* Better UI design
* Real-world dataset integration
* Personality insights dashboard
* Deploy using Docker/AWS

---

## Author

**Harsh Sachan**

* B.Tech CSE (AI Specialization)
* Galgotias College of Engineering and Technology
* LinkedIn: [https://www.linkedin.com/in/harsh-sachan-b3b24a2a2](https://www.linkedin.com/in/harsh-sachan-b3b24a2a2)

---

⭐ If you like this project, give it a star on GitHub.
