# main.py - Адаптированная версия для Android
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.utils import platform
import cv2
import numpy as np
from collections import deque
import time
from datetime import datetime
from ultralytics import YOLO
import threading

class StillnessDetector:
    # ... (тот же код, что и в вашем оригинале)
    pass

class SecurityMonitorApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.model = None
        self.running = False
        self.stillness_detector = None
        self.fps = 30
        self.alert_count = 0
        
    def build(self):
        # Настройки для Android
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
        
        # UI Layout
        layout = BoxLayout(orientation='vertical')
        
        # Кнопки управления
        button_layout = BoxLayout(size_hint=(1, 0.1))
        self.start_btn = Button(text='Запуск мониторинга')
        self.start_btn.bind(on_press=self.start_monitoring)
        self.stop_btn = Button(text='Остановить', disabled=True)
        self.stop_btn.bind(on_press=self.stop_monitoring)
        button_layout.add_widget(self.start_btn)
        button_layout.add_widget(self.stop_btn)
        
        # Статусная строка
        self.status_label = Label(text='Статус: Остановлен', size_hint=(1, 0.05))
        
        # Видео-виджет
        self.image_widget = Image(size_hint=(1, 0.8))
        
        # Информационная панель
        self.info_label = Label(text='Тревог: 0', size_hint=(1, 0.05))
        
        layout.add_widget(button_layout)
        layout.add_widget(self.status_label)
        layout.add_widget(self.image_widget)
        layout.add_widget(self.info_label)
        
        Window.bind(on_request_close=self.on_close)
        
        return layout
    
    def start_monitoring(self, instance):
        # Загрузка модели YOLO (упрощенная версия для Android)
        threading.Thread(target=self.load_model_thread).start()
        
    def load_model_thread(self):
        try:
            self.status_label.text = 'Загрузка модели YOLO...'
            # Для Android используем упрощенную модель
            self.model = YOLO('yolov8n-int8.tflite')  # TensorFlow Lite версия
            self.start_camera()
        except Exception as e:
            self.status_label.text = f'Ошибка: {str(e)}'
    
    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.status_label.text = 'Ошибка: Не удалось открыть камеру'
            return
        
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 30
            
        self.stillness_detector = StillnessDetector(
            fps=self.fps,
            stillness_threshold=8.0,
            alert_after_seconds=10.0
        )
        
        self.running = True
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.status_label.text = 'Статус: Мониторинг активен'
        
        Clock.schedule_interval(self.update_frame, 1.0/self.fps)
    
    def update_frame(self, dt):
        if not self.running or self.capture is None:
            return
            
        ret, frame = self.capture.read()
        if not ret:
            return
        
        # Зеркальное отражение для front camera
        frame = cv2.flip(frame, 1)
        
        # Обработка через YOLO (упрощенно для производительности)
        try:
            results = self.model(frame, verbose=False)[0]
            
            if results.boxes is not None:
                classes = results.boxes.cls.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
                
                for class_id, box in zip(classes, boxes):
                    if int(class_id) == 0:  # Person class
                        x1, y1, x2, y2 = box
                        center = ((x1 + x2)//2, (y1 + y2)//2)
                        
                        # Рисуем bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Информация о неподвижности (упрощенно)
                        cv2.putText(frame, "Person Detected", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        except Exception as e:
            pass
        
        # Отображение статуса
        alert_count = len(self.stillness_detector.active_alerts) if self.stillness_detector else 0
        self.info_label.text = f'Тревог: {alert_count} | FPS: {self.fps}'
        
        if alert_count > 0:
            self.status_label.text = '!!! ВНИМАНИЕ: Обнаружена неподвижность !!!'
            self.status_label.color = (1, 0, 0, 1)
        else:
            self.status_label.text = 'Статус: Нормально'
            self.status_label.color = (0, 1, 0, 1)
        
        # Конвертация для отображения в Kivy
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture
    
    def stop_monitoring(self, instance):
        self.running = False
        if self.capture:
            self.capture.release()
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.status_label.text = 'Статус: Остановлен'
        self.status_label.color = (1, 1, 1, 1)
        Clock.unschedule(self.update_frame)
    
    def on_close(self, *args):
        self.stop_monitoring(None)
        return True

if __name__ == '__main__':
    SecurityMonitorApp().run()