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

st.set_page_config(
    page_title="Drug Consumption Prediction",
    page_icon="💊"
)
st.title('💊 Drug Consumption Prediction')

with st.container(border=True):
    st.subheader('Demographic')
    col1, col2 = st.columns(2)
    with col1:
        age_options = list(age_mapping.keys())
        age = st.selectbox('Age', age_options)

        ethnicity_options = list(ethnicity_mapping.keys())
        ethnicity = st.selectbox('Ethnicity', ethnicity_options)

        education_options = list(education_mapping.keys())
        education = st.selectbox('Education Level', education_options)

    with col2:
        gender_options = list(gender_mapping.keys())
        gender = st.selectbox('Gender', gender_options)

        country_options = list(country_mapping.keys())
        country = st.selectbox('Country', country_options)

with st.container(border=True):
    st.subheader('Psychological')
    col3, col4 = st.columns(2)
    with col3:
        nscore = st.number_input('Neuroticism', min_value=12, max_value=60, value=12, step=1)
        oscore = st.number_input('Openness', min_value=12, max_value=60, value=12, step=1)
        cscore = st.number_input('Conscientiousness', min_value=12, max_value=60, value=12, step=1)
        ss = st.number_input('SS', min_value=12, max_value=60, value=12, step=1)

    with col4:
        escore = st.number_input('Extraversion', min_value=12, max_value=60, value=12, step=1)
        ascore = st.number_input('Agreeableness', min_value=12, max_value=60, value=12, step=1)
        impulsive = st.number_input('Impulsive', min_value=12, max_value=60, value=12, step=1)

if st.button('Predict'):
    try:
        model = xgb.XGBRegressor()
        model.load_model('model.xgb')

        age_encoded = age_mapping[age]
        gender_encoded = gender_mapping[gender]
        education_encoded = education_mapping[education]
        country_encoded = country_mapping[country]
        ethnicity_encoded = ethnicity_mapping[ethnicity]

        input_data = np.array([[age_encoded, gender_encoded, education_encoded, country_encoded, ethnicity_encoded,
                                nscore, escore, oscore, ascore, cscore, impulsive, ss]])
        drug_col = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
        y_test_pred = model.predict(input_data)[0]
        y_test_pred = np.round(y_test_pred).astype(int)

        st.write(f"The predicted drug consumption is: {drug_col[y_test_pred]}")
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)
            st.write(f'The probability of the predicted drug consumption is: {probability}')

    except Exception as e:
        st.error(f'Error making prediction: {e}')
