# Skincare Recommendations

![skincare-101-800x445](https://github.com/user-attachments/assets/4f39145b-9ec8-42d1-a90e-4faec3c09a3c)

## Context
Skincare has become increasingly popular in recent years. Regardless of gender, identifying your skin type and 
establishing a personalized skincare routine is crucial for maintaining healthy, radiant skin. A tailored approach 
can significantly enhance your appearance and overall skin health.

## Objectives
- Create a AI that identifies skin types and recommends products tailored to specific skincare needs.
- Understand which skincare products have the greatest impact on sales and why.

## Datasets
The datasets used in this project were sourced from Kaggle with implementation from chatgpt to make them more complete:
 - Oily, Dry and Normal Skin Types Dataset: https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset
 - Amazon Skincare Products: https://www.kaggle.com/datasets/namantrisoliya/amazon-skincare-products

## Results
The initial results were not as strong as expected, even after experimenting with various models. The validation accuracy fell short 
of what is considered efficient. After multiple tests, it became evident that the primary issue lies within the data itself. This is 
something that will be addressed in future iterations. You can try out the current version of the app, where you can upload an image 
to determine your skin type: Online Skin Type Classifier. Once the data and code are refined, the user experience and accuracy should 
improve.

## Pipeline:
![pipeline_skins](https://github.com/user-attachments/assets/c04bc24f-18d0-4454-aebb-178a636f7c67)

## Machine Learning Algorithm
The model architecture includes one rescaling layer, one padding layer, one reduction layer, and two hidden layers (the first with 6 
filters, and the second with 128 neurons).

![model_ml](https://github.com/user-attachments/assets/bce8677d-31d6-4d9e-9b6b-c14e207f1ff5)

## Process
After downloading the necessary datasets, I began exploring the data using Visual Studio Code. I realized that I needed additional 
libraries that I hadn’t worked with before, so I searched for online resources and tutorials to guide me through the challenges I 
encountered.

For my machine learning model, I consulted various videos, articles, and documentation to deepen my understanding. I experimented 
with the tools and eventually published the results on a Streamlit application.

Once the model was complete, I turned my attention to the Amazon skincare products dataset to extract insights. Below is one example 
of the insights I discovered:


![grafico](https://github.com/user-attachments/assets/cb2163ae-e276-4d5d-9948-474fff2762a8)

## References

- Image Processing with OpenCV and Python. Availabel at: https://youtu.be/kSqxn6zGE0c?si=wUtJW8aOctj2ypTp
- Reconhecimento de Imagens com Deep Learning usando TensorFlow e Keras // Mão no Código by Red Hat #8. Availabel at: https://youtu.be/7MItgjXU3_E?si=KCwV-pdFG8JI4EcL
- Build a Deep CNN Image Classifier with ANY Images. Availabel at:  https://youtu.be/jztwpsIzEGc?si=NdN1BZckRoBdCC5W
- Scikit-learn documentation. Availabel at: https://scikit-learn.org/stable/modules/preprocessing.html
- TensorFlow documentation Availabel at: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
- Keras documentation. Availabel at: https://keras.io/2.15/api/layers/pooling_layers/max_pooling2d/
- Alura Course - Keras: primeiros passos em redes neurais. Availabel at: https://cursos.alura.com.br/course/keras-primeiros-passos-redes-neurais
- Alura Course - TensorFlow Keras: classificando imagens com redes convolucionais, Availabel at: https://cursos.alura.com.br/course/tensorflow-keras-classificando-imagens-redes-convolucionais
- Book - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
- Book - Storytelling with data
