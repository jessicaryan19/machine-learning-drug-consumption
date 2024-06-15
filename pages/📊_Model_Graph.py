import streamlit as st
st.set_page_config(
    page_title="Model Graph",
    page_icon="📊"
)
st.title('📊 Model Graph')

st.subheader('Evaluation Matrix')
st.image('output.png')

st.subheader('Drug Graph Based on Usage')
st.image('drugs.png')

st.subheader('Training Accuracy')
st.image('training_acc.png')

st.subheader('Testing Accuracy')
st.image('testing_acc.png')