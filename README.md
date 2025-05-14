# 🚢 Titanic Survival Predictor

A machine learning web app that predicts whether a passenger would survive the Titanic disaster, based on user-input features like age, gender, class, and fare. Built using **Scikit-learn** and deployed with **Streamlit**.

---

## 📊 About the Project

This project uses the classic [Titanic dataset](https://www.kaggle.com/competitions/titanic) from Kaggle to build a binary classification model to predict survival outcomes.

The model achieves:
- 🟢 **Training Accuracy**: ~80%
- 🟡 **Test Accuracy**: ~78%
- 🏁 **Kaggle Score**: 0.77

---

## 🧠 Features Used

- Passenger Class (`Pclass`)
- Sex (`Sex`)
- Age
- Number of Siblings/Spouses (`SibSp`)
- Number of Parents/Children (`Parch`)
- Fare
- Embarked Port (`Embarked`)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Streamlit (for front end)

---

## 🚀 Try the App

👉 [Live Demo on Streamlit Cloud](#) *(link will go here after deployment)*

---

## 📦 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
