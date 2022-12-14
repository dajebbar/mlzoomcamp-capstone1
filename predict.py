import numpy as np
import streamlit as st
import tensorflow as tf
import efficientnet.keras as efn
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image


model = tf.keras.models.load_model('./kitchenwareModel.h5', compile=False)
labels = {0: 'cup', 1: 'fork', 2: 'glass', 3: 'knife', 4: 'plate', 5: 'spoon'}

def image_processing(img_path):
    """

    """
    img = load_img(img_path,target_size=(224,224,3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img,[0])

    y_softmax = model.predict(img, verbose=0)
    y_preds = np.argmax(y_softmax, axis=-1)

    y = " ".join(str(x) for x in y_preds)
    y = int(y)

    res = labels[y]

    return res

def run():
    """
    """
    banner_img = Image.open("banner.png")
    banner_img = banner_img.resize((900, 250))
    st.image(banner_img, use_column_width=False)
    st.title("DataTalksClub Kitchenware classification Competition")
    st.markdown(
        """
        <h4 style='text-align: left; color: #d73b5c;'>
        Classify kitchen items into 6 categories: 
        </h4>
        <ul>
        <li>cups</li>
        <li>glasses</li>
        <li>plates</li>
        <li>spoons</li>
        <li>forks</li>
        <li>knives</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    img_input = st.file_uploader("Choose your image", type=["jpg", "png"] )

    if img_input is not None:
        img = Image.open(img_input)
        img = img.resize((224, 224))
        st.image(img, use_column_width=False)
        save_img_path = img_input.name

        with open(save_img_path, "wb") as f:
            f.write(img_input.getbuffer())
        
        if st.button("Predict"):
            result = image_processing(save_img_path)
            st.success(f"The kitchen utensil provided is:   {result.upper()}")

run()

