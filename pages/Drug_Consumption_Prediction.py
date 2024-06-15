import streamlit as st
import numpy as np
import sklearn
import xgboost as xgb

# Define mappings
age_mapping = {
    '18 - 24': -0.95197,
    '25 - 34': -0.07854,
    '35 - 44': 0.49788,
    '45 - 54': 1.09449,
    '55 - 64': 1.82213,
    '65+': 2.59171
}

gender_mapping = {
    'Female': 0.48246,
    'Male': -0.48246
}

education_mapping = {
    'Left School Before 16 years': -2.43591,
    'Left School at 16 years': -1.73790,
    'Left School at 17 years': -1.43719,
    'Left School at 18 years': -1.22751,
    'Some College,No Certificate Or Degree': -0.61113,
    'Professional Certificate/Diploma': -0.05921,
    'University Degree': 0.45468,
    'Masters Degree': 1.16365,
    'Doctorate Degree': 1.98437
}

country_mapping = {
    'Australia': -0.09765,
    'Canada': 0.24923,
    'New Zealand': -0.46841,
    'Other': -0.28519,
    'Republic of Ireland': 0.21128,
    'UK': 0.96082,
    'USA': -0.57009
}

ethnicity_mapping = {
    'Asian': -0.50212,
    'Black': -1.10702,
    'Mixed-Black/Asian': 1.90725,
    'Mixed-White/Asian': 0.12600,
    'Mixed-White/Black': -0.22166,
    'Other': 0.11440,
    'White': -0.31685
}

# Title
st.set_page_config(
    page_title="Drug Consumption Prediction",
    page_icon="📈"
)
st.title('Drug Consumption Prediction')

# Debugging information
st.write("Debug Info: Checking age_mapping contents...")
st.write(age_mapping)

try:
    age_options = list(age_mapping.keys())
    st.write("Debug Info: age_options successfully created")
    age = st.selectbox('Age', age_options)
except AttributeError as e:
    st.error(f"AttributeError encountered: {e}")
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"AttributeError: {e}\n")
    st.stop()

# Gender options
gender_options = list(gender_mapping.keys())
# Gender
gender = st.selectbox('Gender', gender_options)

# Education options
education_options = list(education_mapping.keys())
# Education
education = st.selectbox('Education Level', education_options)

# Country options
country_options = list(country_mapping.keys())
# Country
country = st.selectbox('Country', country_options)

# Ethnicity options
ethnicity_options = list(ethnicity_mapping.keys())
# Ethnicity
ethnicity = st.selectbox('Ethnicity', ethnicity_options)

# Nscore
nscore = st.number_input('Nscore', min_value=0, max_value=100, value=0, step=1)

# Escore
escore = st.number_input('Escore', min_value=0, max_value=100, value=0, step=1)

# Oscore
oscore = st.number_input('Oscore', min_value=0, max_value=100, value=0, step=1)

# Ascore
ascore = st.number_input('Ascore', min_value=0, max_value=100, value=0, step=1)

# Cscore
cscore = st.number_input('Cscore', min_value=0, max_value=100, value=0, step=1)

# Impulsive
impulsive = st.number_input('Impulsive', min_value=0, max_value=100, value=0, step=1)

# SS
ss = st.number_input('SS', min_value=0, max_value=100, value=0, step=1)

# Prediction button
if st.button('Predict'):
    try:
        # Load the model
        model = xgb.XGBRegressor()
        model.load_model('model.xgb')

        # Encode categorical inputs using mappings
        age_encoded = age_mapping[age]
        gender_encoded = gender_mapping[gender]
        education_encoded = education_mapping[education]
        country_encoded = country_mapping[country]
        ethnicity_encoded = ethnicity_mapping[ethnicity]

        # Prepare input data for prediction (both categorical and numerical features)
        input_data = np.array([[age_encoded, gender_encoded, education_encoded, country_encoded, ethnicity_encoded,
                                nscore, escore, oscore, ascore, cscore, impulsive, ss]])

        # Make prediction
        drug_col = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

        y_test_pred = model.predict(input_data)[0]
        y_test_pred = np.round(y_test_pred).astype(int)

        st.write(f"The predicted drug consumption is: {drug_col[y_test_pred]}")

        # (Optional) If your model outputs probabilities

        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)
            st.write(f'The probability of the predicted drug consumption is: {probability}')

    except Exception as e:
        st.error(f'Error making prediction: {e}')
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Prediction Error: {e}\n")
