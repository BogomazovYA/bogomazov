import sys
import cv2
import numpy as np
import os
from datetime import datetime

# Библиотеки интерфейса
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QListWidget, QGroupBox, QTextEdit
)
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import QTimer, Qt

# Библиотеки для распознавания и рисования
import face_recognition
from PIL import Image, ImageDraw, ImageFont

class BiometricSystem(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Система биометрической идентификации (Диплом)")
        self.resize(1300, 850)

        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #ffffff; font-family: Arial; font-size: 14px; }
            QPushButton { background-color: #0078d7; padding: 10px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #0063b1; }
            QPushButton:disabled { background-color: #555; }
            QPushButton#reset { background-color: #d70000; } 
            QListWidget, QTextEdit { background-color: #1e1e1e; border: 1px solid #555; border-radius: 4px; }
            QGroupBox { border: 1px solid #555; margin-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { font-size: 14px; }
        """)

        # Переменные
        self.users = {}
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_frame = None
        
        # --- НОВАЯ ПЕРЕМЕННАЯ: КОГО ИЩЕМ КОНКРЕТНО ---
        self.target_user = None # Если None - ищем всех. Если имя - ищем только его.

        # Шрифты
        try:
            self.font_main = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            self.font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except:
            self.font_main = ImageFont.load_default()
            self.font_bold = ImageFont.load_default()

        self.build_ui()
        self.load_users()

        self.camera_active = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_camera_frame)

    def load_users(self):
        if not os.path.exists("users"): os.makedirs("users")
        if not os.path.exists("snapshots"): os.makedirs("snapshots")

        self.users = {}
        self.known_face_encodings = []
        self.known_face_names = []

        self.log_to_console("Инициализация базы данных...")
        
        for filename in os.listdir("users"):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join("users", filename)
                try:
                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img)
                    if len(encodings) > 0:
                        name = os.path.splitext(filename)[0]
                        self.users[name] = encodings[0]
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                except Exception:
                    pass
        
        self.log_to_console(f"База загружена. Пользователей: {len(self.users)}")
        self.refresh_user_list()

    def build_ui(self):
        # === ЛЕВАЯ ПАНЕЛЬ ===
        self.list_users = QListWidget()
        self.list_users.itemClicked.connect(self.select_user_from_list) # <-- КЛИК ПО СПИСКУ

        self.btn_reset_selection = QPushButton("❌ Сброс выбора")
        self.btn_reset_selection.setObjectName("reset")
        self.btn_reset_selection.clicked.connect(self.reset_selection)
        self.btn_reset_selection.setEnabled(False)

        self.btn_add_user = QPushButton("➕ Добавить лицо")
        self.btn_add_user.clicked.connect(self.add_user)
        self.btn_refresh = QPushButton("🔄 Перезагрузить БД")
        self.btn_refresh.clicked.connect(self.reload_db)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Выберите человека для проверки:"))
        left_layout.addWidget(self.list_users)
        left_layout.addWidget(self.btn_reset_selection) # Кнопка сброса
        left_layout.addWidget(self.btn_add_user)
        left_layout.addWidget(self.btn_refresh)
        
        left_box = QGroupBox("База эталонов")
        left_box.setLayout(left_layout)
        left_box.setFixedWidth(280)

        # === ЦЕНТР ===
        # Статус режима (Авто или Верификация)
        self.mode_label = QLabel("РЕЖИМ: АВТОМАТИЧЕСКИЙ ПОИСК")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet("font-weight: bold; color: #00ff00; font-size: 16px; margin-bottom: 5px;")

        self.image_label = QLabel("СИСТЕМА ГОТОВА К РАБОТЕ")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #555; background-color: #000; color: #888;")
        self.image_label.setFixedSize(640, 480)

        self.btn_start_cam = QPushButton("▶ ЗАПУСК СИСТЕМЫ")
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_snapshot = QPushButton("📸 ФИКСАЦИЯ")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.btn_snapshot.setEnabled(False)
        self.btn_stop_cam = QPushButton("⏹ ОСТАНОВКА")
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_stop_cam.setEnabled(False)

        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.btn_start_cam)
        cam_layout.addWidget(self.btn_snapshot)
        cam_layout.addWidget(self.btn_stop_cam)

        center_layout = QVBoxLayout()
        center_layout.addWidget(self.mode_label) # Добавил метку режима
        center_layout.addStretch()
        center_layout.addWidget(self.image_label)
        center_layout.addLayout(cam_layout)
        center_layout.addStretch()
        center_box = QGroupBox("Терминал наблюдения")
        center_box.setLayout(center_layout)

        # === ПРАВАЯ ПАНЕЛЬ ===
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.logs)
        right_box = QGroupBox("Системный журнал")
        right_box.setLayout(right_layout)
        right_box.setFixedWidth(300)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_box)
        main_layout.addWidget(center_box)
        main_layout.addWidget(right_box)
        self.setLayout(main_layout)

    def select_user_from_list(self, item):
        """Когда кликнули по имени в списке"""
        name = item.text().replace("👤 ", "")
        self.target_user = name
        self.btn_reset_selection.setEnabled(True)
        
        # Визуальное оповещение
        self.mode_label.setText(f"РЕЖИМ: ПРОВЕРКА [{name.upper()}]")
        self.mode_label.setStyleSheet("font-weight: bold; color: #00aaff; font-size: 16px;")
        self.log_to_console(f"Включен режим верификации для: {name}")

    def reset_selection(self):
        """Сброс в автоматический режим"""
        self.target_user = None
        self.list_users.clearSelection()
        self.btn_reset_selection.setEnabled(False)
        
        self.mode_label.setText("РЕЖИМ: АВТОМАТИЧЕСКИЙ ПОИСК")
        self.mode_label.setStyleSheet("font-weight: bold; color: #00ff00; font-size: 16px;")
        self.log_to_console("Включен режим автоматического поиска")

    def log_to_console(self, text):
        time = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'logs'):
            self.logs.append(f"[{time}] {text}")

    def refresh_user_list(self):
        self.list_users.clear()
        for name in self.users.keys():
            self.list_users.addItem(f"👤 {name}")

    def reload_db(self):
        self.load_users()
        self.log_to_console("База данных обновлена вручную.")

    def add_user(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Фото сотрудника", "", "Images (*.jpg *.png)")
        if not file_path: return
        save_path, _ = QFileDialog.getSaveFileName(self, "Имя сотрудника", "./users/User.jpg", "Images (*.jpg)")
        if not save_path: return
        
        img = cv2.imread(file_path)
        if img is not None:
            cv2.imwrite(save_path, img)
            self.log_to_console(f"Добавлен новый эталон: {os.path.basename(save_path)}")
            self.reload_db()

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): return
            self.camera_active = True
            self.timer.start(30)
            self.btn_start_cam.setEnabled(False)
            self.btn_stop_cam.setEnabled(True)
            self.btn_snapshot.setEnabled(True)
            self.log_to_console("Видеозахват активирован.")

    def stop_camera(self):
        if self.camera_active:
            self.camera_active = False
            self.timer.stop()
            if self.cap: self.cap.release()
            self.image_label.clear()
            self.image_label.setText("СИСТЕМА ОСТАНОВЛЕНА")
            self.btn_start_cam.setEnabled(True)
            self.btn_stop_cam.setEnabled(False)
            self.btn_snapshot.setEnabled(False)
            self.log_to_console("Видеозахват завершен.")

    def take_snapshot(self):
        if self.last_frame is not None:
            name = f"snapshots/evidence_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(name, self.last_frame)
            self.log_to_console(f"Факт зафиксирован: {name}")
            QMessageBox.information(self, "Фиксация", "Снимок с метаданными сохранен.")

    def process_camera_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # PIL Рисование
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4; right *= 4; bottom *= 4; left *= 4

            # --- ЛОГИКА РЕЖИМОВ ---
            name = "НЕИЗВЕСТНЫЙ"
            color = (255, 0, 0) # Красный
            status_text = "ОТКАЗ В ДОСТУПЕ"
            confidence = 0.0

            if self.target_user:
                # === РЕЖИМ 1: ВЕРИФИКАЦИЯ (Проверка конкретного человека) ===
                if self.target_user in self.users:
                    # Сравниваем ТОЛЬКО с выбранным
                    target_encoding = self.users[self.target_user]
                    # compare_faces возвращает список [True] или [False]
                    match = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.6)[0]
                    dist = face_recognition.face_distance([target_encoding], face_encoding)[0]
                    
                    if match:
                        name = self.target_user
                        color = (0, 255, 0) # Зеленый
                        status_text = "ЛИЧНОСТЬ ПОДТВЕРЖДЕНА"
                        confidence = (1 - dist)
                    else:
                        # Если лицо есть, но не то, которое мы ждем
                        name = "НЕСОВПАДЕНИЕ"
                        color = (255, 0, 0) # Красный
                        status_text = f"ОЖИДАЛСЯ: {self.target_user.upper()}"
                        confidence = 0.0
            else:
                # === РЕЖИМ 2: ИДЕНТИФИКАЦИЯ (Автопоиск среди всех) ===
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                dists = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(dists) > 0:
                    best_idx = np.argmin(dists)
                    if matches[best_idx]:
                        name = self.known_face_names[best_idx]
                        color = (0, 255, 0)
                        status_text = "СОВПАДЕНИЕ"
                        confidence = (1 - dists[best_idx])

            if confidence > 1.0: confidence = 1.0

            # --- ОТРИСОВКА ---
            # Рамка
            draw.rectangle([left, top, right, bottom], outline=color, width=3)

            # Верхняя плашка (Имя)
            draw.rectangle([left, top - 40, right, top], fill=(0, 0, 0, 180))
            draw.text((left + 5, top - 35), name, font=self.font_main, fill=(255, 255, 255))

            # Нижняя плашка (Статус)
            draw.rectangle([left, bottom, right, bottom + 60], fill=(0, 0, 0, 180))
            draw.text((left + 5, bottom + 5), status_text, font=self.font_bold, fill=color)

            # Шкала
            if color == (0, 255, 0): # Рисуем шкалу только если все хорошо
                bar_width = right - left - 10
                fill_width = int(bar_width * confidence)
                draw.rectangle([left + 5, bottom + 40, right - 5, bottom + 50], fill=(100, 100, 100))
                draw.rectangle([left + 5, bottom + 40, left + 5 + fill_width, bottom + 50], fill=color)
                draw.text((right - 50, bottom + 35), f"{int(confidence*100)}%", font=self.font_main, fill=(255, 255, 255))

        # Возврат в OpenCV
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self.last_frame = frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(), aspectMode=Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BiometricSystem()
    window.show()
    sys.exit(app.exec())