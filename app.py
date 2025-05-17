import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset (make sure you have it loaded similarly in your app)
@st.cache_data
def load_data():
    return pd.read_csv('dataset/titanic.csv')  # or your actual dataset path/source

df = load_data()

# Title and Overview
st.markdown('<h1 style="text-align: center;">🚢 Titanic Survival Predictor</div>', unsafe_allow_html=True)

st.markdown("## Dataset Overview")
total_passengers = len(df)
survived_count = df['Survived'].sum()
not_survived_count = total_passengers - survived_count
survival_rate = survived_count / total_passengers * 100

st.write(f"Total Passengers: {total_passengers}")
st.write(f"Survived: {survived_count}")
st.write(f"Did Not Survive: {not_survived_count}")
st.write(f"Survival Rate: {survival_rate:.2f}%")

st.subheader("🧾 Dataset Feature Description")

# Create a DataFrame with feature descriptions
feature_desc = {
    "Feature": [
        "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    ],
    "Description": [
        "Unique ID for each passenger",
        "Survival (0 = No, 1 = Yes)",
        "Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)",
        "Passenger's full name",
        "Passenger's sex",
        "Passenger's age in years",
        " number of siblings/spouses aboard",
        "number of parents/children aboard",
        "Ticket number",
        "Fare paid for ticket",
        "Cabin number",
        "Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)"
    ]
}

feature_df = pd.DataFrame(feature_desc)

# Display the feature table
st.table(feature_df)


# Survival Count Plot
st.markdown("## Survival Count")
fig1, ax1 = plt.subplots()
sns.countplot(x='Survived', data=df, ax=ax1)
ax1.set_xticklabels(['Did Not Survive', 'Survived'])
st.pyplot(fig1)

# Survival by Gender
st.markdown("## Survival by Gender")
fig2, ax2 = plt.subplots()
sns.countplot(x='Sex', hue='Survived', data=df, ax=ax2)
ax2.legend(title='Survived', labels=['No', 'Yes'])
st.pyplot(fig2)

# Survival by Passenger Class
st.markdown("## Survival by Passenger Class")
fig3, ax3 = plt.subplots()
sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax3)
ax3.legend(title='Survived', labels=['No', 'Yes'])
ax3.set_xlabel('Passenger Class')
st.pyplot(fig3)

# Line graph: Survival Rate by Age
st.subheader("📈 Survival Rate by Age (Line Chart)")
age_survival = df.groupby('Age')['Survived'].mean().dropna()

fig, ax = plt.subplots()
ax.plot(age_survival.index, age_survival.values)
ax.set_title('Survival Rate by Age')
ax.set_xlabel('Age')
ax.set_ylabel('Survival Rate')
st.pyplot(fig)

# Add a horizontal rule before prediction UI
st.markdown("---")


# Load the model
model = joblib.load('titanic_model.pkl')

st.title("Check Your Survival Chances 🤔")

st.markdown("Enter passenger information below:")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Convert to model input format
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [0 if sex == 'male' else 1],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [ {'S': 0, 'C': 1, 'Q': 2}[embarked] ]
})

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🟢 The passenger **would survive**.")
    else:
        st.error("🔴 The passenger **would not survive**.")
