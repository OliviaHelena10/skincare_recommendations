import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1S3-lb3WG2pIsfRS93Qyx_cPOXJwbRnnE'
    gdown.download(url, 'quant_model16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='quant_model16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter

def load_image():
    uploaded_file = st.file_uploader('Insert your face picture', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Image was successfully loaded')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def prevision(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image) 
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['dry', 'normal', 'oily']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilities (%)'] = 100 * output_data[0]

    fig = px.bar(df, y='classes', x='probabilities (%)', orientation='h', text='probabilities (%)',
                 title='Probability of your skin type')
    
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Skin type classifier",
        page_icon="💁‍♀️",
    )
    st.write("# Skin type classifier 💁‍♀️ ")
    # Load model
    interpreter = load_model()

    # Load image
    image = load_image()

    if image is not None:
        prevision(interpreter, image)

if __name__ == "__main__":
    main()

