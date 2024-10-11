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

        image = image.resize((640, 640))
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
    
    predicted_class = classes[np.argmax(output_data[0])]
    return predicted_class


def menu():
    st.write('\nChoose your one priority when it comes to your preference')
    st.write('\n\t1. Low price\n\t2. Expensive\n\t3. Brand\n\t4. Specific product \
          \n\t5. Other treatments(acne, sensitive, others)\n\t6. No preferences')
    c = int(input('\n\nYour choice, please digit the equivalent number: '))
    
    return c


def menu_3():
    st.write('\nChoose based on your prefered brand')
    st.write("\n\t[A] Nivea            \n\t[B] Bioderma \
           \n\t[C] Cetaphil         \n\t[D] CeraVe \
           \n\t[E] Neutrogena       \n\t[F] La Roche-Posay \
           \n\t[G] L'Oreal Paris    \n\t[H] Maybelline \
           \n\t[I] Garnier          \n\t[J] Dove")
    b = input('\n\nYour choice, please digit the equivalent letter: ')

    if   b =='A' or b =='a': f = 'Nivea'            
    elif b =='B' or b =='b': f = 'Bioderma'         
    elif b =='C' or b =='c': f = 'Cetaphil'         
    elif b =='D' or b =='d': f = 'CeraVe'           
    elif b =='E' or b =='e': f = 'Neutrogena'       
    elif b =='F' or b =='f': f = 'La Roche-Posay'   
    elif b =='G' or b =='g': f = "L'Oreal Paris"    
    elif b =='H' or b =='h': f = 'Maybelline'       
    elif b =='I' or b =='i': f = 'Garnier'          
    elif b =='J' or b =='j': f = 'Dove'             
    else: print('Not a valid option.')
        
    return f


def menu_4():
    st.write('\nChoose based on your prefered product')
    st.write("\n\t[A] Moisturizer     \n\t[B] Primer \
           \n\t[C] Cleanser        \n\t[D] Sunscreen\
           \n\t[E] Eye cream       \n\t[F] Face Mask \
           \n\t[G] Facial spray    \n\t[H] Toner \
           \n\t[I] Serum")
    b = input('\n\nYour choice, please digit the equivalent letter: ')
    
    if   b =='A' or b =='a': f = 'Moisturizer'
    elif b =='B' or b =='b': f = 'Primer'
    elif b =='C' or b =='c': f = 'Cleanser'
    elif b =='D' or b =='d': f = 'Sunscreen'
    elif b =='E' or b =='e': f = 'Eye cream'
    elif b =='F' or b =='f': f = 'Face Mask'
    elif b =='G' or b =='g': f = 'Facial sprays'
    elif b =='H' or b =='h': f = 'Toner'
    elif b =='I' or b =='i': f = 'Serum'
    else: print('Not a valid option.')
    
    return f


def menu_5():
    st.write('\nChoose based on your prefered specificity')
    st.write("\n\t[A] Acne           \n\t[B] Sensitive \
           \n\t[C] Cracked        \n\t[D] Dull \
           \n\t[E] Rough          \n\t[F] All")
    b = input('\n\nYour choice, please digit the equivalent letter: ')
    
    if   b =='A' or b == 'a': f = 'Acne'  
    elif b =='B' or b == 'b': f = 'Sensitive'      
    elif b =='C' or b == 'c': f = 'Cracked'       
    elif b =='D' or b == 'd': f = 'Dull'      
    elif b =='E' or b == 'e': f = 'Rough'    
    elif b =='F' or b == 'f': f = 'All'        
    else: print('Not a valid option.')         
    
    return f


def contains(str, col_list):
    return any(str in item for item in col_list)


def deep_cases(c, df):
    if c == 3: 
        output = menu_3()
        result = df[df['Brand'].apply(lambda x: output in x)]
        return result
                
    if c == 4:
        output = menu_4()
        result = df[df['Product'].apply(lambda x: output in x)]
        
    if c == 5: 
        output = menu_5()
        result = df[df['Skin_Type'].apply(lambda col_lista: contains(output, col_lista))]
    
    return result


def priorize(c, result):
    
    if   c == 1: result = result[result['Price'].apply(lambda x: 400 > x)]       # Returns all the cheap prices
    elif c == 2: result = result[result['Price'].apply(lambda x: 400 < x)]       # Returns all the expensive prices 
    elif c == 3 or c == 4 or c == 5: result = deep_cases(c, result)
    elif c == 6: ...                                                             # Returns all the products
    else:
        print('Invalid choice.')
        menu()
            
    return result

def gerar_relatorio_txt(result_df, file_name):
    # Defining the max lenght
    max_len = {
        'Title': max(result_df['Title'].apply(len).max(), len("Title")),
        'Product': max(result_df['Product'].apply(len).max(), len("Product")),
        'Brand': max(result_df['Brand'].apply(len).max(), len("Brand")),
        'Skin_Type': max(result_df['Skin_Type'].apply(lambda lista: len(", ".join(lista))).max(), len("Skin Type")),
        'Price': max(len(f"${price:.2f}") for price in result_df['Price']),
        'Link': max(result_df['Link'].apply(len).max(), len("Link")),
    }
    
    # Calculates columns width
    total_width = sum(max_len.values()) + 5 * 5
    
    with open(file_name, 'w', encoding='utf-8') as f:
        # Title
        f.write("Relatório de Produtos de Skincare\n")
        f.write("=" * total_width + "\n\n") 

        # Dinamic columns name with dinamic space
        f.write(f"{'Title':<{max_len['Title']}} | {'Product':<{max_len['Product']}} | {'Brand':<{max_len['Brand']}} | {'Skin Type':<{max_len['Skin_Type']}} | {'Price':<{max_len['Price']}} | {'Link':<{max_len['Link']}}\n")
        f.write("=" * total_width + "\n")
        
        # Adding values
        for index, row in result_df.iterrows():
            title = row['Title']
            product = row['Product']
            brand = row['Brand']
            skin_types = ", ".join(row['Skin_Type'])  # Transform list to string
            price = f"${row['Price']:.2f}"            # Formats price
            link = row['Link']
            
            # Dinamic spaces
            f.write(f"{title:<{max_len['Title']}} | {product:<{max_len['Product']}} | {brand:<{max_len['Brand']}} | {skin_types:<{max_len['Skin_Type']}} | {price:<{max_len['Price']}} | {link:<{max_len['Link']}}\n")

    print(f"Report saved as {file_name}")
    
    
def main():
    df = pd.read_csv('C:/Users/fatima/Documents/programs/Projetos/Skin_types/clean_products.csv')
  
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
        
    result = priorize(menu(), df)
    gerar_relatorio_txt(result, "my_skincare_report.txt")

if __name__ == "__main__":
    main()