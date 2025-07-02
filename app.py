import cv2
import numpy as np
import threading
import queue
import time
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceBlurApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Blur Streaming")
        
        # Инициализация детектора лиц
        self.face_detector = self.load_face_detector()
        
        # Пул потоков
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Переменные
        self.setup_variables()
        self.setup_queues()
        
        # Интерфейс
        self.create_widgets()
        
        # Проверка зависимостей
        self.check_dependencies()

    def load_face_detector(self):
        # Пытаемся загрузить DNN модель
        prototxt = "deploy.prototxt"
        caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            try:
                net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                logger.info("Loaded DNN face detector")
                return net
            except Exception as e:
                logger.error(f"Failed to load DNN: {e}")
        
        # Fallback на каскады Хаара
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("Using Haar cascade detector")
        return cascade

    def setup_variables(self):
        self.input_source = tk.StringVar(value="webcam")
        self.rtsp_url = tk.StringVar(value="")
        self.media_file = tk.StringVar()
        self.blur_enabled = tk.BooleanVar(value=True)
        self.blur_type = tk.StringVar(value="gaussian")
        self.blur_intensity = tk.IntVar(value=30)
        self.detection_confidence = tk.IntVar(value=70)
        self.min_face_size = tk.IntVar(value=20)  # Минимальный размер лица (дальние)
        self.max_face_size = tk.IntVar(value=400)  # Максимальный размер лица (близкие)
        self.output_rtsp_enabled = tk.BooleanVar(value=True)
        self.output_rtsp_port = tk.StringVar(value="8554")
        self.output_stream_name = tk.StringVar(value="stream")
        
        self.streaming = False
        self.cap = None
        self.ffmpeg_process = None
        self.frame_count = 0
        self.use_dnn = isinstance(self.face_detector, cv2.dnn_Net)

    def setup_queues(self):
        self.rtsp_queue = queue.Queue(maxsize=2)
        self.gui_queue = queue.Queue(maxsize=2)
        self.last_gui_update = 0
        self.last_frame = None

    def create_widgets(self):
        # Input Source Frame
        input_frame = ttk.LabelFrame(self.root, text="Input Source")
        input_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Radiobutton(input_frame, text="Web Camera", variable=self.input_source, value="webcam").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(input_frame, text="RTSP Stream", variable=self.input_source, value="rtsp").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(input_frame, text="Media File", variable=self.input_source, value="file").grid(row=2, column=0, sticky="w")
        
        self.rtsp_entry = ttk.Entry(input_frame, textvariable=self.rtsp_url, width=40)
        self.rtsp_entry.grid(row=1, column=1, padx=5, pady=2)
        
        self.file_entry = ttk.Entry(input_frame, textvariable=self.media_file, width=40)
        self.file_entry.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Button(input_frame, text="Browse", command=self.browse_file).grid(row=2, column=2, padx=5)
        
        # Detection Settings Frame
        detection_frame = ttk.LabelFrame(self.root, text="Detection Settings")
        detection_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Checkbutton(detection_frame, text="Use DNN Detector", variable=self.use_dnn, 
                      state="disabled").grid(row=0, column=0, sticky="w")
        
        ttk.Label(detection_frame, text="Confidence Threshold:").grid(row=1, column=0, sticky="w")
        ttk.Scale(detection_frame, from_=50, to=95, variable=self.detection_confidence, 
                 orient="horizontal").grid(row=1, column=1, sticky="we", pady=2)
        
        ttk.Label(detection_frame, text="Min Face Size (far):").grid(row=2, column=0, sticky="w")
        ttk.Scale(detection_frame, from_=10, to=100, variable=self.min_face_size,
                 orient="horizontal").grid(row=2, column=1, sticky="we", pady=2)
        
        ttk.Label(detection_frame, text="Max Face Size (near):").grid(row=3, column=0, sticky="w")
        ttk.Scale(detection_frame, from_=100, to=800, variable=self.max_face_size,
                 orient="horizontal").grid(row=3, column=1, sticky="we", pady=2)
        
        # Blur Settings Frame
        blur_frame = ttk.LabelFrame(self.root, text="Blur Settings")
        blur_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Checkbutton(blur_frame, text="Enable Face Blur", variable=self.blur_enabled).grid(row=0, column=0, sticky="w", columnspan=2)
        
        ttk.Label(blur_frame, text="Blur Type:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(blur_frame, textvariable=self.blur_type, 
                    values=["gaussian", "pixel", "black", "circle"], 
                    state="readonly").grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(blur_frame, text="Blur Intensity:").grid(row=2, column=0, sticky="w")
        ttk.Scale(blur_frame, from_=1, to=100, variable=self.blur_intensity, 
                 orient="horizontal").grid(row=2, column=1, sticky="we", pady=2)
        
        # Output RTSP Frame
        output_frame = ttk.LabelFrame(self.root, text="Output RTSP Stream")
        output_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Checkbutton(output_frame, text="Enable RTSP Output", variable=self.output_rtsp_enabled).grid(row=0, column=0, sticky="w", columnspan=2)
        
        ttk.Label(output_frame, text="Port:").grid(row=1, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_rtsp_port, width=10).grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(output_frame, text="Stream Name:").grid(row=2, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_stream_name, width=20).grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(output_frame, text="Output URL:").grid(row=3, column=0, sticky="w")
        self.output_url_label = ttk.Label(output_frame, text="rtsp://localhost:8554/stream")
        self.output_url_label.grid(row=3, column=1, sticky="w", pady=2)
        
        # Control Buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(padx=10, pady=10, fill="x")
        
        self.start_button = ttk.Button(control_frame, text="Start Streaming", command=self.toggle_streaming)
        self.start_button.pack(side="left", padx=5)
        
        # Video Display
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)
        
        # Trace variables
        self.output_rtsp_port.trace_add("write", self.update_output_url)
        self.output_stream_name.trace_add("write", self.update_output_url)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")])
        if filename:
            self.media_file.set(filename)
    
    def update_output_url(self, *args):
        port = self.output_rtsp_port.get()
        name = self.output_stream_name.get()
        self.output_url_label.config(text=f"rtsp://localhost:{port}/{name}")
    
    def check_dependencies(self):
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.ffmpeg_available = True
        except:
            self.ffmpeg_available = False
            messagebox.showwarning("Warning", 
                                 "FFmpeg is not installed. RTSP output will not work.")

    def detect_faces(self, frame):
        faces = []
        (h, w) = frame.shape[:2]
        
        if self.use_dnn and isinstance(self.face_detector, cv2.dnn_Net):
            # DNN detection
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                       (300, 300), (104.0, 177.0, 123.0))
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > (self.detection_confidence.get() / 100):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face_width = endX - startX
                    
                    if (face_width >= self.min_face_size.get() and 
                        face_width <= self.max_face_size.get()):
                        faces.append((startX, startY, face_width, endY-startY))
        else:
            # Haar cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size.get(), self.min_face_size.get()),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in detected:
                if w <= self.max_face_size.get():
                    faces.append((x, y, w, h))
        
        return faces

    def apply_blur(self, frame, faces):
        for (x, y, w, h) in faces:
            # Увеличиваем область для маленьких лиц (дальних)
            if w < 50:
                expansion = int(50 - w)
                x = max(0, x - expansion)
                y = max(0, y - expansion)
                w = min(frame.shape[1] - x, w + 2*expansion)
                h = min(frame.shape[0] - y, h + 2*expansion)
            
            if self.blur_type.get() == "gaussian":
                intensity = max(3, self.blur_intensity.get())
                if intensity % 2 == 0:
                    intensity += 1
                
                # Усиливаем размытие для дальних лиц
                if w < 80:
                    intensity = min(51, intensity * 2)
                
                face_region = frame[y:y+h, x:x+w]
                blurred = cv2.GaussianBlur(face_region, (intensity, intensity), 0)
                
                # Плавные границы
                mask = np.zeros((h, w, 3), dtype=np.float32)
                cv2.rectangle(mask, (5, 5), (w-5, h-5), (1, 1, 1), -1)
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                
                frame[y:y+h, x:x+w] = (blurred * mask + face_region * (1 - mask)).astype(np.uint8)
            
            elif self.blur_type.get() == "pixel":
                intensity = max(2, self.blur_intensity.get() // 10)
                if w < 80:  # Увеличиваем пикселизацию для дальних лиц
                    intensity = max(1, intensity // 2)
                
                face_region = frame[y:y+h, x:x+w]
                small = cv2.resize(face_region, (intensity, intensity), 
                                 interpolation=cv2.INTER_LINEAR)
                blurred = cv2.resize(small, (w, h), 
                                  interpolation=cv2.INTER_NEAREST)
                frame[y:y+h, x:x+w] = blurred
            
            elif self.blur_type.get() == "black":
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)
            
            elif self.blur_type.get() == "circle":
                intensity = max(3, self.blur_intensity.get())
                if intensity % 2 == 0:
                    intensity += 1
                
                center = (x + w//2, y + h//2)
                radius = min(w, h) // 2
                
                # Увеличиваем радиус для маленьких лиц
                if w < 80:
                    radius = radius * 1.5
                
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                cv2.circle(mask, center, int(radius), 1, -1)
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                mask = np.dstack([mask]*3)
                
                blurred_frame = cv2.GaussianBlur(frame.copy(), (intensity, intensity), 0)
                frame = (blurred_frame * mask + frame * (1 - mask)).astype(np.uint8)
        
        return frame

    def process_frame(self, frame):
        try:
            # Изменяем размер для обработки
            frame = cv2.resize(frame, (640, 480))
            
            # Обнаруживаем лица с учетом фильтра по размеру
            faces = self.detect_faces(frame)
            
            if faces and self.blur_enabled.get():
                frame = self.apply_blur(frame, faces)
            
            return frame
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def video_capture_thread(self):
        while self.streaming and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from source")
                break
            
            # Обработка кадра
            processed = self.process_frame(frame)
            
            # Отправка в RTSP поток
            if self.output_rtsp_enabled.get() and self.ffmpeg_available:
                try:
                    self.rtsp_queue.put_nowait(processed.copy())
                except queue.Full:
                    pass
            
            # Отправка в GUI
            try:
                self.gui_queue.put_nowait(processed.copy())
            except queue.Full:
                pass
            
            # Контроль частоты кадров
            time.sleep(0.033)  # ~30 FPS
        
        self.stop_streaming()

    def rtsp_stream_thread(self):
        while self.streaming and self.output_rtsp_enabled.get() and self.ffmpeg_available:
            try:
                frame = self.rtsp_queue.get(timeout=1)
                if self.ffmpeg_process and not self.ffmpeg_process.stdin.closed:
                    self.ffmpeg_process.stdin.write(frame.tobytes())
                    self.ffmpeg_process.stdin.flush()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in RTSP thread: {e}")
                break

    def update_gui(self):
        if not self.streaming:
            return
            
        try:
            # Ограничиваем частоту обновления GUI
            if time.time() - self.last_gui_update < 0.033:
                self.root.after(30, self.update_gui)
                return
                
            frame = self.gui_queue.get_nowait()
            
            # Конвертируем кадр для отображения
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Обновляем GUI
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgk)
            
            self.last_gui_update = time.time()
            
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error updating GUI: {e}")
            
        self.root.after(30, self.update_gui)

    def start_rtsp_server(self):
        port = self.output_rtsp_port.get()
        stream_name = self.output_stream_name.get()
        
        command = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', '640x480',
            '-r', '25',
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            '-muxdelay', '0.1',
            f'rtsp://localhost:{port}/{stream_name}'
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info("RTSP server started")
        except Exception as e:
            logger.error(f"Failed to start RTSP server: {e}")
            messagebox.showerror("Error", f"Failed to start RTSP server: {str(e)}")

    def toggle_streaming(self):
        if not self.streaming:
            self.start_streaming()
        else:
            self.stop_streaming()
    
    def start_streaming(self):
        self.streaming = True
        self.start_button.config(text="Stop Streaming")
        self.frame_count = 0
        
        # Открываем видео источник
        input_type = self.input_source.get()
        
        try:
            if input_type == "webcam":
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 25)
            elif input_type == "rtsp":
                self.cap = cv2.VideoCapture(self.rtsp_url.get())
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif input_type == "file":
                self.cap = cv2.VideoCapture(self.media_file.get())
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video source!")
                self.stop_streaming()
                return
            
            # Запускаем RTSP сервер если нужно
            if self.output_rtsp_enabled.get() and self.ffmpeg_available:
                self.start_rtsp_server()
            
            # Запускаем поток захвата видео
            self.video_thread = threading.Thread(target=self.video_capture_thread, daemon=True)
            self.video_thread.start()
            
            # Запускаем RTSP поток если нужно
            if self.output_rtsp_enabled.get() and self.ffmpeg_available:
                self.rtsp_thread = threading.Thread(target=self.rtsp_stream_thread, daemon=True)
                self.rtsp_thread.start()
            
            # Запускаем обновление GUI
            self.update_gui()
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            messagebox.showerror("Error", f"Failed to start streaming: {str(e)}")
            self.stop_streaming()

    def stop_streaming(self):
        self.streaming = False
        self.start_button.config(text="Start Streaming")
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                pass
            self.ffmpeg_process = None
        
        # Очищаем очереди
        while not self.rtsp_queue.empty():
            try:
                self.rtsp_queue.get_nowait()
            except:
                break
                
        while not self.gui_queue.empty():
            try:
                self.gui_queue.get_nowait()
            except:
                break

    def on_closing(self):
        self.stop_streaming()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceBlurApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()