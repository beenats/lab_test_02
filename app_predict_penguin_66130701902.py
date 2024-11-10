 
 
import streamlit as st
import pickle
import pandas as pd
 
# Load the model and encoders
with open('model_penguin_66130701921.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)
 
# Streamlit interface
st.title('Penguin Species Prediction')
 
# Collecting user input for the prediction
island = st.selectbox('Island', island_encoder.classes_)
culmen_length = st.number_input('Culmen Length (mm)', min_value=0.0, max_value=100.0, value=37.0, step=0.1)
culmen_depth = st.number_input('Culmen Depth (mm)', min_value=0.0, max_value=100.0, value=19.3, step=0.1)
flipper_length = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=250.0, value=192.3, step=0.1)
body_mass = st.number_input('Body Mass (g)', min_value=0.0, max_value=8000.0, value=3750.0, step=10.0)
sex = st.selectbox('Sex',  ['MALE','FEMALE'])
 
# Preparing input for prediction
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length],
    'culmen_depth_mm': [culmen_depth],
    'flipper_length_mm': [flipper_length],
    'body_mass_g': [body_mass],
    'sex': [sex]
})
 
# Transforming categorical features using the encoders
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])
 
# Make prediction
y_pred_new = model.predict(x_new)
 
# Reverse transformation to get the species name
result = species_encoder.inverse_transform(y_pred_new)
 
# Display the result
st.write(f'Predicted Species: {result[0]}')
 
 
 
 
