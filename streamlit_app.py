import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

def count_and_percent(sizes):
    def inner(val):
        count = int(round(val * sum(sizes) / 100))
        return f'{count} ({val:.1f}%)'
    return inner


model = joblib.load('xgb_pokemon_model.pkl')
df = pd.read_csv('Pokemon.csv')


def visualization():
    st.header('Data Visualization')
    st.divider()

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Stat Distributions")

    feature = st.selectbox("Select a Stat", df.select_dtypes(include=['int64', 'float64']).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("🥧 Legendary Pokémon Distribution")
    legend_counts = df['Legendary'].value_counts()
    labels = ['Not Legendary', 'Legendary']  
    sizes = legend_counts.sort_index() 
    colors = ['skyblue', 'gold']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct=count_and_percent(sizes), startangle=90)
    ax.axis('equal') 
    st.pyplot(fig)

    st.subheader("🧮 Feature Correlation")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.drop(columns=['Name', '#'], inplace=True)
    df['Type 2'] = df['Type 2'].fillna('None')
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def predict():
    st.title("🧠 Predict Legendary Pokémon")
    st.divider()

    hp = st.slider("HP", 0, 200, 80)
    attack = st.slider("Attack", 0, 200, 80)
    defense = st.slider("Defense", 0, 200, 80)
    sp_atk = st.slider("Sp. Atk", 0, 200, 80)
    sp_def = st.slider("Sp. Def", 0, 200, 80)
    speed = st.slider("Speed", 0, 200, 80)

    features = [[hp, attack, defense, speed, sp_atk, sp_def]]

    if st.button("Predict"):
        prediction = model.predict(features)
        
        if prediction[0]:
            st.success(f"🔥 Predicted: Legendary!")
        else:
            st.info(f"🟢 Predicted: Not Legendary")

def about():
    st.title("🔮Legendary Pokémon Predictor")
    st.markdown("""
                Welcome to the **Legendary Pokémon Predictor** app! 🧠  
                This tool allows you to:

                - 🔍 Explore and visualize Pokémon dataset statistics  
                - 🪄 Predict whether a Pokémon is **Legendary** based on its battle stats  
                - 🎯 Review the model's performance using key evaluation metrics""")

    st.divider()

    st.header("About the Model")
    st.markdown("""
                    This machine learning model was trained on the classic **Pokémon dataset** to predict whether a Pokémon is **Legendary** or not based solely on its battle statistics.

                    ### ✨ Model Overview
                    - **Algorithm:** [XGBoost Classifier](https://xgboost.readthedocs.io/en/stable/)  
                    - **Learning type:** Supervised Binary Classification  
                    - **Use case:** Classify Pokémon into `Legendary` or `Not Legendary`

                    ### 📊 Features Used
                    The model uses six numerical attributes that define a Pokémon’s battle performance:

                    - `HP` – Health Points  
                    - `Attack` – Physical attack power  
                    - `Defense` – Physical damage resistance  
                    - `Sp. Atk` – Special (non-physical) attack power  
                    - `Sp. Def` – Special (non-physical) defense  
                    - `Speed` – Determines who attacks first in battles

                    > No categorical features (like Type or Generation) were used to keep the model stat-based and simplified.

                    ### 🎯 Model Performance

                    The following metrics are based on **5-fold cross-validation** across the dataset:
                    """)
    st.divider()

    X = df[['HP', 'Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def']]
    y = df['Legendary']

    y_pred = cross_val_predict(model, X, y, cv=5)

    report = classification_report(y, y_pred)

    st.header("📋 Classification Report")
    st.code(report, language='text')
    st.divider()

    st.header("📊 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(['Not Legendary', 'Legendary'])
    ax.set_yticklabels(['Not Legendary', 'Legendary'])
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    st.divider()

pg = st.navigation([
  st.Page(about, title="Welcome to the app!"),
  st.Page(visualization, title="Dataset Visualization"),
  st.Page(predict, title="Let's Predict"),
])
pg.run()


