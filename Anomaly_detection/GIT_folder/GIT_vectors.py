import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch




# --- Конфигурация ---
MODEL_NAME = "microsoft/git-large-coco"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# --- Функции ---
def extract_image_info(directory_path):
    image_files = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, filename)) and filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return image_files


def VLM_vectors(img_path):
    """Возвращает векторное представление изображения"""
    try:
        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            vision_outputs = model.git.image_encoder(
                pixel_values=inputs.pixel_values,
                output_hidden_states=True
            )
            last_hidden_state = vision_outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :]
            visual_projection = model.git.visual_projection(cls_embedding)
        
        return visual_projection.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Ошибка обработки {img_path}: {str(e)}")
        return None


def process_images(image_paths):
    """Обрабатывает изображения и выводит прогресс"""
    vectors = []
    total = len(image_paths)

    for index, path in enumerate(image_paths):
        vector = VLM_vectors(path)
        if vector is not None:
            vectors.append(vector)

        if (index + 1) % 1 == 0:
            progress_msg = f"\rОбработано {index + 1}/{total} изображений"
            print(progress_msg, end='', flush=True)

    print() 
    return vectors


def GIT_vectors_func(directory):

    glob = {'image_paths': []}

    glob['image_paths'] = extract_image_info(directory)
    result_vectors = np.array(process_images(glob['image_paths']))

    print(f"Получено векторов: {len(result_vectors)}")
    print(f"Размерность вектора: {result_vectors.shape[1] if len(result_vectors) > 0 else 'N/A'}")

    return glob, result_vectors
