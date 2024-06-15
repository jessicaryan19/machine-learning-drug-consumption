import streamlit as st
import numpy as np
import xgboost as xgb

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
    page_icon="📈"
)
st.title('Drug Consumption Prediction')

age_options = list(age_mapping.keys())
age = st.selectbox('Age', age_options)

gender_options = list(gender_mapping.keys())
gender = st.selectbox('Gender', gender_options)

education_options = list(education_mapping.keys())
education = st.selectbox('Education Level', education_options)

country_options = list(country_mapping.keys())
country = st.selectbox('Country', country_options)

ethnicity_options = list(ethnicity_mapping.keys())
ethnicity = st.selectbox('Ethnicity', ethnicity_options)