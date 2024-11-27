# import csv
# import tkinter as tk
# import numpy as np
# from Service import recognition
# from Service.CreateDataset import transformation_array
# from .NeuroNet import NeuralNetworkWindow
# import os
# from dotenv import load_dotenv
# from PIL import Image
# import io
#
# class DrawingApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Приложение для рисования")
#         self.master.geometry("400x300")
#
#         # Панель для кнопок
#         button_frame = tk.Frame(self.master)
#         button_frame.pack(side=tk.TOP, fill=tk.X)
#
#         # Кнопки
#         self.recognition_button = tk.Button(button_frame, text="Распознать", command=self.picture_recognition)
#         self.recognition_button.pack(side=tk.LEFT, padx=5)
#
#         self.clear_button = tk.Button(button_frame, text="Очистить", command=self.clear_canvas)
#         self.clear_button.pack(side=tk.LEFT, padx=5)
#
#         self.neural_network_button = tk.Button(button_frame, text="Нейронная сеть", command=self.open_neural_network_window)
#         self.neural_network_button.pack(side=tk.LEFT, padx=5)
#
#         # Настройка холста
#         self.canvas = tk.Canvas(self.master, width=200, height=200, bg='white')
#         self.canvas.pack(pady=16)
#
#         self.canvas.bind("<B1-Motion>", self.paint)
#
#         # Текстовое поле для вывода
#         self.output_text = tk.StringVar()
#         self.output_text.set("")  # Начальная пустая строка
#         font = ('Helvetica', 10)  # Пример размера шрифта
#         self.output_field = tk.Entry(self.master, textvariable=self.output_text, state='disabled', width=40, font=font)
#         self.output_field.pack(pady=5)
#
#     def paint(self, event):
#         x1, y1 = (event.x - 1), (event.y - 1)
#         x2, y2 = (event.x + 1), (event.y + 1)
#         self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
#
#     def picture_recognition(self):
#         # Получаем постскрипт-данные с холста
#         ps = self.canvas.postscript(colormode='color')
#
#         # Используем PIL для открытия и сохранения изображения
#         img = Image.open(io.BytesIO(ps.encode('utf-8')))
#         img = img.convert("RGB")  # Преобразуем в RGB, если необходимо
#
#         load_dotenv()
#         const_width = int(os.getenv('const_width'))
#
#         img = img.resize((const_width, const_width))
#         # Сохраняем в BMP
#         path_name = "Files/drawing"
#         img.save(path_name + ".bmp", "BMP")
#
#         img_gray = img.convert('L')
#
#         result = []
#         result.append(transformation_array(np.array(img_gray)))
#
#         with open(path_name + ".csv", 'w', newline='') as file:
#             writer = csv.writer(file, delimiter=';')
#             writer.writerow(result[0])
#
#         result = recognition(path_name + ".csv")
#
#         self.output_text.set(f"Результат: {result}!")
#
#     def clear_canvas(self):
#         self.canvas.delete("all")
#
#     def open_neural_network_window(self):
#         NeuralNetworkWindow(self.master)



import csv
import tkinter as tk
from tkinter import messagebox, Toplevel
import numpy as np
from Service import recognition, train, validation
from Service.CreateDataset import transformation_array
from .Metric import MetricWindow
from PIL import Image
import os
import io
from dotenv import load_dotenv


class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Нейронная сеть")
        self.master.geometry("900x320")

        # === Левый блок: Рисование ===
        self.drawing_frame = tk.LabelFrame(self.master, text="Рисование", padx=10, pady=10)
        self.drawing_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.canvas = tk.Canvas(self.drawing_frame, width=200, height=200, bg="white")
        self.canvas.pack(pady=16)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.recognition_button = tk.Button(self.drawing_frame, text="Распознать", command=self.picture_recognition)
        self.recognition_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.drawing_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Поле для вывода результата
        self.output_text = tk.StringVar()
        self.output_text.set("")
        self.output_field = tk.Entry(self.drawing_frame, textvariable=self.output_text, state='disabled', width=40)
        self.output_field.pack(pady=5)

        # === Верхний блок: параметры обучения ===
        self.params_frame = tk.LabelFrame(self.master, text="Параметры обучения", padx=10, pady=10)
        self.params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(self.params_frame, text="Эпохи:").pack(side=tk.LEFT)
        self.epochs_entry = tk.Entry(self.params_frame)
        self.epochs_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(self.params_frame, text="Скорость обучения:").pack(side=tk.LEFT)
        self.speed_entry = tk.Entry(self.params_frame)
        self.speed_entry.pack(side=tk.LEFT, padx=5)

        # === Левый блок: опции обучения ===
        self.train_frame = tk.LabelFrame(self.master, text="Обучение", padx=10, pady=10)
        self.train_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)

        self.learn_new_button = tk.Button(self.train_frame, text="Обучить с нуля", command=self.train_from_scratch)
        self.learn_new_button.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        self.retrain_button = tk.Button(self.train_frame, text="Дообучить", command=self.retrain)
        self.retrain_button.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        # === Правый блок: валидация и метрики ===
        self.metrics_frame = tk.LabelFrame(self.master, text="Валидация и Метрики", padx=10, pady=10)
        self.metrics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=5)

        self.validate_button = tk.Button(self.metrics_frame, text="Протестировать", command=self.validate_model)
        self.validate_button.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        self.train_metrics_button = tk.Button(self.metrics_frame, text="Графики обучения", command=self.show_graphs_for_train)
        self.train_metrics_button.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        self.validation_metrics_button = tk.Button(self.metrics_frame, text="Графики валидации", command=self.show_graphs_for_validation)
        self.validation_metrics_button.pack(side=tk.TOP, fill='x', padx=5, pady=5)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    def picture_recognition(self):
        ps = self.canvas.postscript(colormode="color")

        # Используем PIL для конвертации изображения
        img = Image.open(io.BytesIO(ps.encode("utf-8")))
        img = img.convert("RGB")

        load_dotenv()
        const_width = int(os.getenv("const_width", 28))  # Пример значения по умолчанию
        img = img.resize((const_width, const_width))

        # Сохраняем изображение
        path_name = "Files/drawing"
        img.save(path_name + ".bmp", "BMP")

        img_gray = img.convert("L")
        result = []
        result.append(transformation_array(np.array(img_gray)))

        with open(path_name + ".csv", "w", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(result[0])

        result = recognition(path_name + ".csv")
        self.output_text.set(f"Результат: {result}!")

    def clear_canvas(self):
        self.canvas.delete("all")

    def train_from_scratch(self):
        if self.confirm_action("Вы уверены, что хотите начать обучение с нуля?"):
            epochs = self.get_epochs()
            speed = self.get_speed()
            if epochs is not None:
                train(speed, epochs, True)

    def retrain(self):
        if self.confirm_action("Вы уверены, что хотите начать дообучение?"):
            epochs = self.get_epochs()
            speed = self.get_speed()
            if epochs is not None:
                train(speed, epochs, False)

    def validate_model(self):
        epochs = self.get_epochs()
        validation(epochs)

    def get_speed(self):
        try:
            speed = float(self.speed_entry.get())
            return speed
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное значение скорости!")
            return None

    def get_epochs(self):
        try:
            epochs = int(self.epochs_entry.get())
            return epochs
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число эпох!")
            return None

    @staticmethod
    def confirm_action(message):
        return messagebox.askyesno("Подтверждение действия", message)

    def show_graphs_for_train(self):
        MetricWindow(self.master, "train_metrics")

    def show_graphs_for_validation(self):
        MetricWindow(self.master, "validate_metrics")


