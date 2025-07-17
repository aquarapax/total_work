import os
import cv2
import torch
import numpy as np
import json
import gradio as gr
from torchvision import transforms
from torchvision.models.video import r3d_18
import torch.nn as nn

class Config:
    CLIP_LEN = 16  # Количество кадров в клипе
    FRAME_SIZE = 112  # Размер кадра
    CLASSES_FILE = "ucf101_classes.json"

def load_classes():
    """Загрузка классов"""
    if not os.path.exists(Config.CLASSES_FILE):
        raise FileNotFoundError(f"Файл классов не найден: {Config.CLASSES_FILE}")
    
    with open(Config.CLASSES_FILE, "r", encoding="utf-8") as f:
        class_data = json.load(f)
    
    return class_data["original"], class_data["russian"], class_data["mapping"]

def load_model():
    """Загрузка модели"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Архитектура
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 101)
    
    # Загрузка весов
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device).eval()
    
    return model, device

def process_video(video_path, model, device, transform):
    """Обработка видео и получение предсказания"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    # Общее количество кадров
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, total_frames // Config.CLIP_LEN)
    
    model_frames = []
    for i in range(Config.CLIP_LEN * skip):
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip == 0:  # Равномерная выборка кадров
            # Преобразование кадра
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (Config.FRAME_SIZE, Config.FRAME_SIZE))
            model_frames.append(transform(frame_rgb))
            if len(model_frames) == Config.CLIP_LEN:
                break
    
    cap.release()
    
    # Дублирование последнего кадра до длинны
    while len(model_frames) < Config.CLIP_LEN:
        model_frames.append(model_frames[-1] if model_frames else torch.zeros(3, Config.FRAME_SIZE, Config.FRAME_SIZE))
    
    # Подготовка данных для модели
    clip = torch.stack(model_frames).unsqueeze(0).to(device)
    clip = clip.permute(0, 2, 1, 3, 4) 
    
    # Получение предсказания
    with torch.no_grad():
        output = model(clip)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probabilities, 1)
    
    return pred_idx.item(), max_prob.item()

def analyze_video(video):
    """Анализ видео"""
    try:
        # Загрузка модели и классов
        model, device = load_model()
        original_classes, russian_classes, class_mapping = load_classes()
        
        # Трансформации
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Обработка видео и предсказание
        pred_idx, prob = process_video(video, model, device, transform)
        pred_class = original_classes[pred_idx]
        russian_name = class_mapping.get(pred_class, "N/A")
        
        return f"{pred_class} ({russian_name}) - Вероятность: {prob:.2%}"
        
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Интерфейс градио
def create_interface():
    # Примеры видео
    examples = [
        ["vid1.mp4"],
        ["vid2.mp4"],
        ["vid3.mp4"],
        ["vid4.mp4"],
        ["vid5.mp4"],
        ["vid6.mp4"],
        ["vid7.mp4"],
        ["vid8.mp4"],
        ["vid9.mp4"],
        ["vid10.mp4"]
    ]
    
    # Создание интерфейса
    with gr.Blocks(title="Обработка видео", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Обработка видео")
        gr.Markdown("Загрузите видео для определения действия")
        
        with gr.Row():
            # Колонка
            with gr.Column():
                video_input = gr.Video(label="Видео", height=480, autoplay=True)
                label_output = gr.Textbox(label="Результат")
                upload_button = gr.UploadButton("Загрузить видео", file_types=["video"])
                
                # Примеры видео
                gr.Markdown("### Примеры")
                with gr.Column(elem_classes="vertical-examples"):
                    gr.Examples(
                        examples=examples,
                        inputs=video_input,
                        outputs=label_output,
                        fn=analyze_video,
                        label='',
                        examples_per_page=10
                    )
        
        # Обработчики событий
        upload_button.upload(
            fn=lambda file: file.name,
            inputs=upload_button,
            outputs=video_input
        )
        
        video_input.change(
            fn=analyze_video,
            inputs=video_input,
            outputs=label_output
        )
    
    return demo

if __name__ == "__main__":
    print('Для выхода ctrl+C')
    demo = create_interface()
    demo.launch()