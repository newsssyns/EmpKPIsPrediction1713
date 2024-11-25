import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the model and encoders
with open('model_kpi_66130701713.pkl', 'rb') as file:
    model, department_encoder, region_encoder, education_encoder, gender_encoder, recruitment_channel_encoder = pickle.load(file)

# Load the dataset
df = pd.read_csv('Uncleaned_employees_final_dataset.csv').drop('employee_id', axis=1)

# Streamlit App Title
st.title('Employee KPIs Prediction and Visualization App')

# Session state to track tab selection
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Tabs for different functionalities
tabs = ['Predict KPIs', 'Visualize Data', 'Predict from CSV']
selected_tab = st.radio('Select a Tab:', tabs, index=st.session_state.tab_selected)

# Update session state based on tab selection
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

# ---- Tab 1: Predict KPIs ----
if st.session_state.tab_selected == 0:
    st.header('Predict KPIs')

    # User Input Form
    with st.form('User Input Form'):
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

        submitted = st.form_submit_button('Predict')

    if submitted:
        # Create a DataFrame for the input
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

        # Apply encoders
        user_input['department'] = department_encoder.transform(user_input['department'])
        user_input['region'] = region_encoder.transform(user_input['region'])
        user_input['education'] = education_encoder.transform(user_input['education'])
        user_input['gender'] = gender_encoder.transform(user_input['gender'])
        user_input['recruitment_channel'] = recruitment_channel_encoder.transform(user_input['recruitment_channel'])

        # Make a prediction
        prediction = model.predict(user_input)
        st.subheader('Prediction Result:')
        st.write('KPIs_met_more_than_80:', prediction[0])

# ---- Tab 2: Visualize Data ----
elif st.session_state.tab_selected == 1:
    st.header('Visualize Data')

    condition_feature = st.selectbox('Select a Feature for Filtering:', df.columns)
    condition_values = st.multiselect('Select Values:', ['Select All'] + df[condition_feature].unique().tolist())

    # Handle "Select All"
    if 'Select All' in condition_values:
        condition_values = df[condition_feature].unique().tolist()

    # Filter and visualize data
    if condition_values:
        filtered_df = df[df[condition_feature].isin(condition_values)]
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=condition_feature, hue='KPIs_met_more_than_80', data=filtered_df, palette='viridis')
        plt.title(f'Employee Count by {condition_feature}')
        plt.xlabel(condition_feature)
        plt.ylabel('Count')
        st.pyplot(fig)

# ---- Tab 3: Predict from CSV ----
elif st.session_state.tab_selected == 2:
    st.header('Predict KPIs from CSV File')

    uploaded_file = st.file_uploader('Upload a CSV File', type='csv')
    if uploaded_file is not None:
        csv_df_org = pd.read_csv(uploaded_file).dropna()
        csv_df = csv_df_org.drop('employee_id', axis=1)

        # Apply encoders
        csv_df['department'] = department_encoder.transform(csv_df['department'])
        csv_df['region'] = region_encoder.transform(csv_df['region'])
        csv_df['education'] = education_encoder.transform(csv_df['education'])
        csv_df['gender'] = gender_encoder.transform(csv_df['gender'])
        csv_df['recruitment_channel'] = recruitment_channel_encoder.transform(csv_df['recruitment_channel'])

        # Make predictions
        predictions = model.predict(csv_df)
        csv_df_org['KPIs_met_more_than_80'] = predictions

        # Display the predictions
        st.subheader('Predicted Results:')
        st.write(csv_df_org)

        # Visualization
        feature_for_visualization = st.selectbox('Select Feature for Visualization:', csv_df_org.columns)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=feature_for_visualization, hue='KPIs_met_more_than_80', data=csv_df_org, palette='viridis')
        plt.title(f'Prediction Visualization by {feature_for_visualization}')
        plt.xlabel(feature_for_visualization)
        plt.ylabel('Count')
        st.pyplot(fig)
