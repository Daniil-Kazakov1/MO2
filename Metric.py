import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MetricWindow:
    def __init__(self, master, file_named):
        self.master = master
        file_path = f'Files/metrics/{file_named}.csv'
        data = pd.read_csv(file_path, sep=';', header=None,
                           names=['loss', 'accuracy', 'precision', 'recall'])
        self.data = data

        # Показываем графики
        self.show_plots()

    def show_plots(self):
        # Создание нового окна для графиков
        plot_window = Toplevel(self.master)
        plot_window.title("Graphs of Metrics")

        # Создание фигуры с 4 графиками: теперь в виде сетки 2x2
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 строки, 2 столбца

        # Устанавливаем стиль графиков
        plt.style.use('bmh')

        # Loss (Верхний левый график)
        axs[0, 0].plot(self.data.index, self.data['loss'], label='Loss',
                       color='navy', linestyle='-', marker='o')
        axs[0, 0].set_title('Loss over Epochs', fontsize=10)
        axs[0, 0].set_xlabel('Epochs', fontsize=9)
        axs[0, 0].set_ylabel('Loss', fontsize=9)
        axs[0, 0].legend(fontsize=8)
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)

        # Accuracy (Верхний правый график)
        axs[0, 1].plot(self.data.index, self.data['accuracy'], label='Accuracy',
                       color='forestgreen', linestyle='-', marker='o')
        axs[0, 1].set_title('Accuracy over Epochs', fontsize=10)
        axs[0, 1].set_xlabel('Epochs', fontsize=9)
        axs[0, 1].set_ylabel('Accuracy', fontsize=9)
        axs[0, 1].legend(fontsize=8)
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)

        # Precision (Нижний левый график)
        axs[1, 0].plot(self.data.index, self.data['precision'], label='Precision',
                       color='darkorange', linestyle='-', marker='o')
        axs[1, 0].set_title('Precision over Epochs', fontsize=10)
        axs[1, 0].set_xlabel('Epochs', fontsize=9)
        axs[1, 0].set_ylabel('Precision', fontsize=9)
        axs[1, 0].legend(fontsize=8)
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)

        # Recall (Нижний правый график)
        axs[1, 1].plot(self.data.index, self.data['recall'], label='Recall',
                       color='crimson', linestyle='-', marker='o')
        axs[1, 1].set_title('Recall over Epochs', fontsize=10)
        axs[1, 1].set_xlabel('Epochs', fontsize=9)
        axs[1, 1].set_ylabel('Recall', fontsize=9)
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)

        # Корректируем межграфиковое пространство
        plt.tight_layout()

        # Встраиваем графики в окно Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

        plot_window.mainloop()
