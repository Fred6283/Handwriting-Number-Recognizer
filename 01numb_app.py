# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:51:17 2021

@author: Fred_R
"""

#%% import libraries

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np

import cv2

#%% App 

# Load model. 
numb_recg_model = load_model("model1.h5")

SIZE = 192

# Create a canvas component
st.title("Handwriting Number Recognizer")
st.write("### Write below a number:" )

canvas_result = st_canvas(
    fill_color = "#ffffff",
    stroke_width=10,
    stroke_color="#ffffff",
    background_color="#000000",
    height=150, width=150,
    drawing_mode="freedraw",
    key="canvas",
    )

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28,28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)
    
if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_test = (test_x.reshape(1,28,28))/255
    x_test_app = x_test.reshape(len(x_test), 28*28)
    pred = numb_recg_model.predict(x_test_app)
    pred_label = np.argmax(pred)
    st.write(f'## Result: {pred_label}')
    st.bar_chart(pred[0])
