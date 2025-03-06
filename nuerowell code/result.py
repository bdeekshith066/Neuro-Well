"""
Given a patient's game, cognitive, motor, and voice scores, the trained model predicts a mobility score (1-10).
Based on the predicted mobility score, the system classifies the patient's condition:
Score < 4 → Requires an exoskeleton.
Score 4-6 → Needs rehab and electrical impulses.
Score > 6 → Needs light exercises for full recovery.

"""

import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set up Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("neurowell-5b8eaaee5d15.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1lRSSzi6IfIEEKkfpMj67QI3S4ukyKzr2_yDlVEFGQcc/edit?gid=0#gid=0").sheet1

def plot_pie_chart(score):
    fig, ax = plt.subplots()
    labels = ['Score', 'Remaining']
    colors = ['#66b3ff', 'lightgray']
    sizes = [score, 10 - score]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Score (Pie Chart)')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig

# Generate a dataset with additional features
np.random.seed(42)
num_samples = 1000
game_scores = np.random.randint(0, 11, (num_samples, 4))

# Assume some dummy cognitive, motor, and voice scores for the sake of the model
cognitive_scores = np.random.randint(1, 11, num_samples)
motor_scores = np.random.randint(1, 11, num_samples)
voice_scores = np.random.randint(1, 11, num_samples)

mobility_scores = []
for i in range(num_samples):
    avg_game_score = np.mean(game_scores[i])
    avg_ability_score = np.mean([cognitive_scores[i], motor_scores[i], voice_scores[i]])
    if avg_game_score >= 8 and avg_ability_score >= 8:
        mobility_score = np.random.randint(8, 11)
    elif avg_game_score >= 5 and avg_ability_score >= 5:
        mobility_score = np.random.randint(5, 8)
    else:
        mobility_score = np.random.randint(1, 5)
    mobility_scores.append(mobility_score)

enhanced_data = {
    "PatientID": np.arange(1, num_samples + 1),
    "Game1_Score": game_scores[:, 0],
    "Game2_Score": game_scores[:, 1],
    "Game3_Score": game_scores[:, 2],
    "Game4_Score": game_scores[:, 3],
    "Cognitive_Score": cognitive_scores,
    "Motor_Score": motor_scores,
    "Voice_Score": voice_scores,
    "Mobility_Score": mobility_scores
}

enhanced_df = pd.DataFrame(enhanced_data)
X = enhanced_df[["Game1_Score", "Game2_Score", "Game3_Score", "Game4_Score", "Cognitive_Score", "Motor_Score", "Voice_Score"]]
y = enhanced_df["Mobility_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(random_state=42)
param_grid_reg = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_reg = GridSearchCV(regressor, param_grid_reg, cv=3, n_jobs=-1, verbose=2)
grid_search_reg.fit(X_train, y_train)
best_regressor = grid_search_reg.best_estimator_

def app():
    st.title("Patient Data Retrieval Portal")
    patient_name = st.text_input("Enter patient name:")

    if st.button("Retrieve Patient Data"):
        all_data = sheet.get_all_records()
        patient_data = [data for data in all_data if data['name'] == patient_name]

        if patient_data:
            st.subheader("Patient Data")
            st.write(patient_data[0])

            speech_score = patient_data[0]['Speech Score']
            snake_score = patient_data[0]['Snake Score']
            ball_score = patient_data[0]['Ball Score']
            emoji_score = patient_data[0]['Emoji Score']

            patient_scores = np.array([[speech_score, snake_score, ball_score, emoji_score, 
                                        np.mean([speech_score, snake_score, ball_score, emoji_score]),
                                        np.mean([speech_score, snake_score, ball_score, emoji_score]),
                                        np.mean([speech_score, snake_score, ball_score, emoji_score])]])

            mobility_prediction = best_regressor.predict(patient_scores)
            mobility_prediction_rounded = np.round(mobility_prediction).astype(int)
            mobility_prediction_rounded = np.clip(mobility_prediction_rounded, 1, 10)

            st.subheader("Prediction and Recommendation")
            recommendation = classify_patient(mobility_prediction_rounded[0])
            st.info(f"""
                **Predicted Mobility Score:** {mobility_prediction_rounded[0]}
                **Recommendation:** {recommendation}

                **Steps to follow:**
                1. Perform regular exercises as recommended by your physiotherapist.
                2. Ensure to take the prescribed medications on time.
                3. Attend all follow-up appointments for continuous assessment.
            """)

            st.subheader("Graphs")
            col1, col2, col3 = st.columns([1, 0.15, 1])

            with col1:
                st.write("Speech Score:")
                fig = plot_pie_chart(speech_score)
                st.pyplot(fig)

                st.write("Snake Score:")
                fig, ax = plt.subplots()
                data = {'Snake Score': [snake_score], 'Remaining': [10 - snake_score]}
                df = pd.DataFrame(data, index=['Score'])
                df.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999'], ax=ax)
                ax.set_title('Snake Score (Stacked Bar Chart)')
                plt.xlabel('')
                plt.ylabel('Score')
                plt.xticks(rotation=0)
                plt.grid(axis='y')
                st.pyplot(fig)

            with col3:
                st.write("Emoji Score:")
                fig, ax = plt.subplots()
                sns.barplot(x=[emoji_score], y=['Emoji Score'], palette='Purples_d', ax=ax, orient='h')
                ax.set_xlim(0, 10)
                ax.set_title('Emoji Score (Horizontal Bar Chart)')
                st.pyplot(fig)

                st.write("Ball Score:")
                fig, ax = plt.subplots()
                ax.pie([ball_score, 10 - ball_score], labels=['Ball Score', 'Remaining'], autopct='%1.1f%%', colors=['green', 'gray'])
                ax.set_title('Ball Score (Pie Chart)')
                st.pyplot(fig)
        else:
            st.write("No data found for the patient.")

def classify_patient(mobility_score):
    if mobility_score < 4:
        return "Person requires exoskeleton"
    elif 4 <= mobility_score <= 6:
        return "Person requires few days of rehab and electrical impulses"
    elif mobility_score > 6:
        return "Person has mobility, requires few days of exercise and physical movement to be completely fine"
    else:
        return "Person's condition needs further assessment"
    



if __name__ == '__main__':
    app()
