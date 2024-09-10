# Skincare Recommendations

![skincare-101-800x445](https://github.com/user-attachments/assets/4f39145b-9ec8-42d1-a90e-4faec3c09a3c)

## Context
Something that is really popular nowadays is skincare. It doesn’t matter if you are a man or a woman, identifying your 
skin type and having the right routine to achieve beautiful skin is an important thing that can make you look better 
and healthier.

## Objectives
- Create a AI that identifies skin types and recommends products specifically for each skin need.
- Understand which skincare products have bigger impact on the sellings.

## Datasets
All the sources that I used were downloaded from Kaggle.
 - Oily, Dry and Normal Skin Types Dataset: https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset
 - Amazon Skincare Products: https://www.kaggle.com/datasets/namantrisoliya/amazon-skincare-products

## Results
Our results weren't the bests even though I tried different models the validation accuracy was inferior then what would be considered
eficcient. After some experiments it obvious that the main problem was on the data so thats something that we are going to fiz later.
Here is the online application where you can send your image to verify your skin type: https://skincarerecommendations-machinelearn.streamlit.app/
After my code is repaired you will probably have a better experience.

## Pipeline:
![pipeline](https://github.com/user-attachments/assets/e18d4ab7-f2f3-49b3-b984-59b760f6633e)

## Machine Learning Algorithm
One reescaling layer, one padding layer, one reducing layer and two hidden layers (the first with 6 filters applied and the other one with 128 neurons).

![model_ml](https://github.com/user-attachments/assets/bce8677d-31d6-4d9e-9b6b-c14e207f1ff5)

## Process
After downloading all the datasets that I needed, I opened VsCode to start investigating my data. I realized that it would
be necessary some other libraries  that I  had never used, so I started searching for some online materials that would help 
me solve some of the problems that I was encountering.

Firstly, for my ML model I found some videos and websites talking about some libraries, did some experiments while accompaning 
tutorials and read the documentations for deeper comprehension.

## References

- Image Processing with OpenCV and Python. Availabel at: https://youtu.be/kSqxn6zGE0c?si=wUtJW8aOctj2ypTp
- Reconhecimento de Imagens com Deep Learning usando TensorFlow e Keras // Mão no Código by Red Hat #8. Availabel at: https://youtu.be/7MItgjXU3_E?si=KCwV-pdFG8JI4EcL
- Build a Deep CNN Image Classifier with ANY Images. Availabel at:  https://youtu.be/jztwpsIzEGc?si=NdN1BZckRoBdCC5W
- Scikit-learn documentation. Availabel at: https://scikit-learn.org/stable/modules/preprocessing.html
- TensorFlow documentation Availabel at: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
- Keras documentation. Availabel at: https://keras.io/2.15/api/layers/pooling_layers/max_pooling2d/
- Alura Course - Keras: primeiros passos em redes neurais. Availabel at: https://cursos.alura.com.br/course/keras-primeiros-passos-redes-neurais
- Alura Course - TensorFlow Keras: classificando imagens com redes convolucionais, Availabel at: https://cursos.alura.com.br/course/tensorflow-keras-classificando-imagens-redes-convolucionais
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
