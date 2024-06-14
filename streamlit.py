# Age,Gender,Education,Country,Ethnicity,Nscore,Escore,Oscore,Ascore,Cscore,Impulsive,SS
# use streamlit to create a web app
# prompt the input of the user

import streamlit as st
from tensorflow.keras.models import load_model

# Title

st.title('Drug Comnsumption Prediction')

# Age
age = st.number_input('Age', min_value=18, max_value=100, value=18, step=1)

# Gender
gender = st.selectbox

# Education
education = st.selectbox

# Country
country = st.selectbox

# Ethnicity
ethnicity = st.selectbox

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

# Button

# Prediction button
if st.button('Predict'):
    try:
        # Load the model
        model = load_model('model.h5')

        # Make prediction
        prediction = model.predict([[age, nscore, escore, oscore, ascore, cscore, impulsive, ss]])
        
        # Output prediction
        st.write(f'The predicted drug consumption is: {prediction}')
        
        # (Optional) If your model outputs probabilities
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba([[age, nscore, escore, oscore, ascore, cscore, impulsive, ss]])
            st.write(f'The probability of the predicted drug consumption is: {probability}')

        # Show model summary
        st.write('Here is the model summary:')
        model.summary(print_fn=lambda x: st.text(x))

    except Exception as e:
        st.error(f'Error making prediction: {e}')