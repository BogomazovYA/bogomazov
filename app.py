import gradio as gr
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# === БЭКЕНД (ЛОГИКА) ===
class BiometricSystem:
    def __init__(self):
        self.users = {}
        self.known_face_encodings = []
        self.known_face_names = []
        
        os.makedirs("users", exist_ok=True)
        
        # Шрифты
        try:
            self.font_main = ImageFont.truetype("arial.ttf", 20)
            self.font_bold = ImageFont.truetype("arial.ttf", 26)
        except:
            self.font_main = ImageFont.load_default()
            self.font_bold = ImageFont.load_default()

        self.reload_database()

    def reload_database(self):
        self.users = {}
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir("users"):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join("users", filename)
                try:
                    img = face_recognition.load_image_file(path)
                    encs = face_recognition.face_encodings(img)
                    if len(encs) > 0:
                        name = os.path.splitext(filename)[0]
                        self.users[name] = encs[0]
                        self.known_face_encodings.append(encs[0])
                        self.known_face_names.append(name)
                except:
                    continue
        return f"База обновлена. Людей: {len(self.users)}"

    def add_user(self, image, name):
        if image is None or not name:
            return "Ошибка: Нет фото или имени"
        save_path = os.path.join("users", f"{name}.jpg")
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        self.reload_database()
        return f"Пользователь {name} добавлен!"

    def process_frame(self, frame):
        if frame is None: return None
        
        # Уменьшаем кадр для скорости (оптимизация)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4; right *= 4; bottom *= 4; left *= 4

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "НЕИЗВЕСТНЫЙ"
            color = (255, 0, 0) # Красный
            status = "ОТКАЗ"
            
            dists = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(dists) > 0:
                best_idx = np.argmin(dists)
                if matches[best_idx]:
                    name = self.known_face_names[best_idx]
                    color = (0, 255, 0) # Зеленый
                    status = "ДОСТУП РАЗРЕШЕН"

            # Рисуем рамку и текст
            draw.rectangle([left, top, right, bottom], outline=color, width=4)
            draw.rectangle([left, top - 40, right, top], fill=(0, 0, 0, 180))
            draw.text((left + 5, top - 35), name, font=self.font_main, fill="white")
            draw.rectangle([left, bottom, right, bottom + 40], fill=(0, 0, 0, 180))
            draw.text((left + 5, bottom + 5), status, font=self.font_bold, fill=color)

        return np.array(pil_img)

system = BiometricSystem()

# === ФРОНТЕНД (ОБНОВЛЕННЫЙ) ===
with gr.Blocks(title="Biometric System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Система биометрической идентификации")
    
    with gr.Tab("📹 Видеонаблюдение"):
        gr.Markdown("Система работает в реальном времени. Просто разрешите доступ к камере.")
        
        # ГЛАВНОЕ ИЗМЕНЕНИЕ: Используем Interface с live=True
        # Это принудительно включает потоковый режим без кнопок "Запись"
        iface = gr.Interface(
            fn=system.process_frame,
            inputs=gr.Image(sources=["webcam"], streaming=True, label="Камера"),
            outputs=gr.Image(label="Результат"),
            live=True,          # <--- ВОТ ЭТО ВКЛЮЧАЕТ АВТО-ОБРАБОТКУ
            flagging_mode="never" # Убираем лишние кнопки
        )

    with gr.Tab("👤 База данных"):
        with gr.Row():
            new_photo = gr.Image(label="Загрузите фото", sources=["upload", "webcam"])
            new_name = gr.Textbox(label="ФИО сотрудника")
        
        btn_add = gr.Button("Сохранить в базу", variant="primary")
        result_msg = gr.Label(label="Статус")
        
        btn_add.click(fn=system.add_user, inputs=[new_photo, new_name], outputs=result_msg)

if __name__ == "__main__":
    demo.launch(share=False)