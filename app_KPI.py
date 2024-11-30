import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model and encoders
try:
    with open('model_kpi_66130701713.pkl', 'rb') as file:
        model, department_encoder, region_encoder, education_encoder, gender_encoder, recruitment_channel_encoder = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model_kpi_66130701713.pkl' not found. Please upload the correct file.")
    st.stop()

# Load dataset
try:
    df = pd.read_csv('Uncleaned_employees_final_dataset.csv')
    df = df.drop('employee_id', axis=1, errors='ignore')
except FileNotFoundError:
    st.warning("Dataset file 'Uncleaned_employees_final_dataset.csv' not found. Some features might not work.")

# Streamlit App
st.title('Employee KPIs App')

# Tab layout
tabs = st.tabs(['Predict KPIs', 'Visualize Data', 'Predict from CSV'])

# Tab 1: Predict KPIs
with tabs[0]:
    st.header('Predict KPIs')

    # User Input Form
    with st.form("input_form"):
        department = st.selectbox('Department', department_encoder.classes_)
        region = st.selectbox('Region', region_encoder.classes_)
        education = st.selectbox('Education', education_encoder.classes_)
        gender = st.radio('Gender', gender_encoder.classes_)
        recruitment_channel = st.selectbox('Recruitment Channel', recruitment_channel_encoder.classes_)
        no_of_trainings = st.slider('Number of Trainings', 1, 10, 1)
        age = st.slider('Age', 18, 60, 30)
        previous_year_rating = st.slider('Previous Year Rating', 1.0, 5.0, 3.0)
        length_of_service = st.slider('Length of Service', 1, 20, 5)
        awards_won = st.checkbox('Awards Won')
        avg_training_score = st.slider('Average Training Score', 40, 100, 70)
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input for prediction
        user_input = pd.DataFrame({
            'department': [department],
            'region': [region],
            'education': [education],
            'gender': [gender],
            'recruitment_channel': [recruitment_channel],
            'no_of_trainings': [no_of_trainings],
            'age': [age],
            'previous_year_rating': [previous_year_rating],
            'length_of_service': [length_of_service],
            'awards_won': [1 if awards_won else 0],
            'avg_training_score': [avg_training_score]
        })

        # Categorical Encoding
        try:
            user_input['department'] = department_encoder.transform(user_input['department'])
            user_input['region'] = region_encoder.transform(user_input['region'])
            user_input['education'] = education_encoder.transform(user_input['education'])
            user_input['gender'] = gender_encoder.transform(user_input['gender'])
            user_input['recruitment_channel'] = recruitment_channel_encoder.transform(user_input['recruitment_channel'])
        except Exception as e:
            st.error(f"Encoding error: {e}")
            st.stop()

        # Predict and display result
        prediction = model.predict(user_input)
        st.subheader('Prediction Result:')
        st.write('KPIs_met_more_than_80:', prediction[0])

# Tab 2: Visualize Data
with tabs[1]:
    st.header('Visualize Data')

    if 'df' in locals():
        condition_feature = st.selectbox('Select Condition Feature:', df.columns)
        condition_values = st.multiselect('Select Condition Values:', ['Select All'] + list(df[condition_feature].unique()))

        if 'Select All' in condition_values:
            condition_values = df[condition_feature].unique()

        if condition_values:
            filtered_df = df[df[condition_feature].isin(condition_values)]
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.countplot(x=condition_feature, hue='KPIs_met_more_than_80', data=filtered_df, palette='viridis')
            plt.title('Number of Employees based on KPIs')
            plt.xlabel(condition_feature)
            plt.ylabel('Number of Employees')
            st.pyplot(fig)
    else:
        st.warning("Dataset not loaded. Please check the file.")

# Tab 3: Predict from CSV
with tabs[2]:
    st.header('Predict from CSV')

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            csv_df = csv_df.drop('employee_id', axis=1, errors='ignore')

            # Encoding
            for col, encoder in zip(
                ['department', 'region', 'education', 'gender', 'recruitment_channel'],
                [department_encoder, region_encoder, education_encoder, gender_encoder, recruitment_channel_encoder]
            ):
                csv_df[col] = encoder.transform(csv_df[col])

            predictions = model.predict(csv_df)
            csv_df['KPIs_met_more_than_80'] = predictions

            # Display results
            st.subheader('Predicted Results:')
            st.write(csv_df)

            # Visualization
            feature = st.selectbox('Select Feature for Visualization:', csv_df.columns)
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.countplot(x=feature, hue='KPIs_met_more_than_80', data=csv_df, palette='viridis')
            plt.title(f'Number of Employees based on KPIs - {feature}')
            plt.xlabel(feature)
            plt.ylabel('Number of Employees')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing file: {e}")
