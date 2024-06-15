import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nbformat
from io import BytesIO

def extract_data_from_ipynb(ipynb_path):
    # Read the notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Execute the cells to get the data
    data = None
    local_vars = {}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            exec(cell.source, globals(), local_vars)
            if 'data' in local_vars:
                data = local_vars['data']
                break
    
    return data

ipynb_path = '../classification.ipynb'

# Extract data from the .ipynb file
data = extract_data_from_ipynb(ipynb_path)

if data is not None:
    # Create a buffer to hold the plot
    buffer = BytesIO()

    # Create the figure and the heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    # Save the plot to the buffer
    plt.savefig(buffer, format='png')

    # Close the plot to free up memory
    plt.close()

    # Seek to the beginning of the buffer
    buffer.seek(0)

    # Display the plot in Streamlit
    st.image(buffer, caption='Heatmap', use_column_width=True)
else:
    st.error("Could not find the data in the provided .ipynb file.")
