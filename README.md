# Advanced Face Blur Streaming App 🎥🛡️

**Интерактивное Python-приложение с GUI для размытия лиц в реальном времени** и трансляции результата через **RTSP-поток**.  
Поддерживаются веб-камеры, RTSP-источники и видеофайлы.  
Используются современные технологии компьютерного зрения: **OpenCV**, **DNN**, **FFmpeg**, **Tkinter**.

---

## 🔗 Демо-видео на YouTube  
👉 [Смотреть демонстрацию FaceBlur Streaming App](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID)

---

## 🧠 Описание проекта

**Advanced Face Blur Streaming App** — это настольное приложение, которое позволяет размывать лица в видео в реальном времени с возможностью настройки фильтров и трансляции результата в RTSP-поток.  
Проект ориентирован на защиту персональных данных в live-трансляциях, видеонаблюдении и медиа.

---

## 🚀 Основные возможности

📡 Поддержка источников:
- Веб-камера  
- RTSP-поток  
- Видео-файл  

🔍 Обнаружение лиц с помощью:
- DNN (ResNet SSD)  
- Haar Cascade (как альтернатива)  

🎛️ Типы размытия лиц:
- Gaussian Blur  
- Pixelation  
- Black Box  
- Circular Blur  

🎚️ Настройки:
- Порог уверенности    
- Интенсивность фильтра  

📤 RTSP-вывод результата (через FFmpeg)  
🖱️ Удобный графический интерфейс (Tkinter)

---

## 🧠 Технологии распознавания лиц

### 1. DNN (Deep Neural Networks от OpenCV)
Модель ResNet SSD, основанная на Caffe:
- 📄 [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)  
- 🧠 [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel)

Пример использования:
```python
cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
```

### 2. Haar Cascade (OpenCV)
Лёгкий, менее ресурсоёмкий метод:
- 📄 [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

Пример использования:
```python
cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```

---

## 🖼️ Скриншоты



---

## ⚙️ Установка

### 🔧 Зависимости

Убедитесь, что установлены:
- Python 3.8+  
- FFmpeg (должен быть доступен в `PATH`)  
  

### 📦 Установка через pip

Клонируйте проект:
```bash
git clone https://github.com/RamazanShimardanov/FaceBlurRealTime
cd FaceBlurRealTime
```

Установите зависимости:
```bash
pip install -r requirements.txt
```

---

## ▶️ Запуск

```bash
python app.py
```

Настройте источник видео, выберите тип фильтра и при необходимости активируйте RTSP-вывод.

---

## 🌐 RTSP-выход

RTSP-поток создаётся локально с помощью FFmpeg. Его можно подключить через OBS Studio, VLC или любой другой RTSP-клиент.

Пример команды для ffmpeg:
```bash
ffmpeg -re -i input.mp4 -f rtsp rtsp://localhost:8554/stream
```

---

## 📚 Лицензия

Проект распространяется под лицензией **MIT**. Свободно используйте и модифицируйте.

---
## Поддержка и сотрудничество
Буду рад поддержке и совместной работе💖:
- Email: yokai1076@gail.com
- Telegram: @gfyly
