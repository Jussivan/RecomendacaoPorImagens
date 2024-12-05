import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def process_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_embedding(image_path):
    processed_image = process_image(image_path)
    return model.predict(processed_image)

image_folder = "images"
image_embeddings = {image_file: extract_embedding(os.path.join(image_folder, image_file)) for image_file in os.listdir(image_folder)}

def find_similar_images(query_image_path, image_embeddings, top_n=5):
    query_embedding = extract_embedding(query_image_path)
    similarities = {image_name: cosine_similarity(query_embedding, embedding)[0][0] for image_name, embedding in image_embeddings.items()}
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

query_image = "query_image.jpg"
results = find_similar_images(query_image, image_embeddings)

plt.figure(figsize=(15, 5))
plt.subplot(1, 6, 1)
plt.title("Query Image")
plt.imshow(load_img(query_image, target_size=(224, 224)))

for i, (image_name, similarity) in enumerate(results):
    image_path = os.path.join(image_folder, image_name)
    plt.subplot(1, 6, i + 2)
    plt.title(f"Sim: {similarity:.2f}")
    plt.imshow(load_img(image_path, target_size=(224, 224)))
