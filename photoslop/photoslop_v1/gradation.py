import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from PIL import Image
from typing import List, Tuple, Optional
from functools import lru_cache


class LinearInterpolation:
    """Класс для линейной интерполяции с кэшированием"""

    def __init__(self, points: List[Tuple[float, float]] = None):
        self.points = points or [(0, 0), (255, 255)]
        self._cache = {}

    def change_points(self, points: List[Tuple[float, float]]):
        """Обновление точек с очисткой кэша"""
        self.points = sorted(points, key=lambda p: p[0])
        self._cache.clear()

    @lru_cache(maxsize=1024)
    def interpolate_single(self, x: float) -> float:
        """Интерполяция для одного значения с кэшированием"""
        points = tuple(self.points)
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if x1 <= x <= x2:
                if x2 == x1:
                    return y1
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        return points[-1][1]

    def interpolate_array(self, arr: np.ndarray) -> np.ndarray:
        """Интерполяция массива значений"""
        if not self.points or len(self.points) < 2:
            return arr

        # Векторизованная интерполяция
        flat = arr.flatten()
        result = np.zeros_like(flat, dtype=np.float32)

        points = sorted(self.points, key=lambda p: p[0])
        x_nodes, y_nodes = zip(*points)
        x_nodes = np.array(x_nodes)
        y_nodes = np.array(y_nodes)

        for i in range(len(x_nodes) - 1):
            mask = (flat >= x_nodes[i]) & (flat <= x_nodes[i + 1])
            if i == 0:
                mask_left = flat < x_nodes[0]
                result[mask_left] = y_nodes[0]

            x1, y1 = x_nodes[i], y_nodes[i]
            x2, y2 = x_nodes[i + 1], y_nodes[i + 1]

            if x2 != x1:
                result[mask] = y1 + (y2 - y1) * (flat[mask] - x1) / (x2 - x1)
            else:
                result[mask] = y1

        mask_right = flat > x_nodes[-1]
        result[mask_right] = y_nodes[-1]

        return result.reshape(arr.shape).clip(0, 255).astype(np.uint8)


class InteractiveInterpolation:
    def __init__(self, initial_points: Optional[List[Tuple[float, float]]] = None,
                 image_path: Optional[str] = None):
        self.points = initial_points.copy() if initial_points else [(0, 0), (255, 255)]

        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.load_image()

        self.interpolation = LinearInterpolation(self.points)

        self.selected_point = None
        self.dragging = False
        self.add_mode = False

        self.setup_figure()

        self.setup_image_display()
        self.setup_plot()

        self.process_and_update_image()
        self.setup_text()
        self.update_plot()

    def load_image(self):
        """Загрузка изображения"""
        if self.image_path:
            try:
                self.original_image = Image.open(self.image_path).convert('RGB')
            except Exception as e:
                print(f"Ошибка загрузки изображения: {e}")
                self.create_placeholder_image()
        else:
            self.create_placeholder_image()

    def create_placeholder_image(self):
        """Создание тестового изображения-заглушки"""
        size = (400, 300)
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        for i in range(size[0]):
            arr[:, i, :] = i * 255 // size[0]

        self.original_image = Image.fromarray(arr)

    def setup_figure(self):
        """Настройка фигуры и компоновки"""
        self.fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(2, 2, width_ratios=[2, 1], figure=self.fig)

        self.ax = self.fig.add_subplot(gs[0,0])
        self.ax_image = self.fig.add_subplot(gs[0, 1])
        self.ax_hist = self.fig.add_subplot(gs[1, 0])
        self.ax_texts = self.fig.add_subplot(gs[1, 1])
        plt.subplots_adjust(bottom=0.15)
        self.btn_ax_reset = plt.axes([0.59, 0.05, 0.1, 0.05])
        self.btn_ax_add = plt.axes([0.7, 0.05, 0.1, 0.05])
        self.btn_ax_clear = plt.axes([0.81, 0.05, 0.1, 0.05])
    def setup_hist(self):
        pass
    def update_hist(self):
        self.ax_hist.clear()
        img_array = np.array(self.processed_image)
        img_lum = np.dot(img_array[..., :3], [0.333, 0.334, 0.333]).astype(np.uint8)
        img_vec = img_lum.reshape(-1)
        values, counts = np.unique(img_vec, return_counts=True)
        self.ax_hist.bar(values, counts, width=0.8, edgecolor='black', alpha=0.7)
        self.ax_hist.set_xlabel('Значения')
        self.ax_hist.set_ylabel('Частота')
        self.ax_hist.set_title('Гистограмма распределения')
        self.ax_hist.grid(axis='y', alpha = 0.3)
    def setup_text(self):
        self.ax_texts.axis('off')
        self.ax_texts.set_xlim(0, 1)
        self.ax_texts.set_ylim(0, 1)
        self.ax_texts.text(0.1, 0.95, 'Информационная панель')
        self.info_texts = {
            'mode': self.ax_texts.text(0.05, 0.8, 'думайте', fontsize=11, va='top', fontweight='bold')
        }
    def update_text(self):
        self.info_texts['mode'].set_text(self.mode_text)
    def setup_image_display(self):
        self.ax_image.axis('off')

        if self.original_image:
            dummy_array = np.array(self.original_image)
        else:
            dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)

        self.image_display = self.ax_image.imshow(dummy_array, vmin=0, vmax=255)
        self.ax_image.set_title(f'Слоп, размер: {dummy_array.shape}', fontsize=12, fontweight='bold')

    def setup_plot(self):
        """Настройка графика и UI элементов"""

        self.btn_reset = Button(self.btn_ax_reset, 'Сброс')
        self.btn_add = Button(self.btn_ax_add, 'Добавить')
        self.btn_clear = Button(self.btn_ax_clear, 'Очистить')
        self.btn_save = Button(self.btn_ax_clear, 'Сохранить')

        self.btn_reset.on_clicked(self.reset_points)
        self.btn_add.on_clicked(self.toggle_add_mode)
        self.btn_clear.on_clicked(self.clear_points)


        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def process_and_update_image(self):
        """Обработка изображения и обновление отображения"""
        if self.original_image is None or len(self.points) < 2:
            if self.original_image is not None:
                self.processed_image = np.array(self.original_image)
            else:
                self.processed_image = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            img_array = np.array(self.original_image)
            processed_channels = []
            for channel in range(3):
                channel_data = img_array[:, :, channel].astype(np.float32)
                processed = self.interpolation.interpolate_array(channel_data)
                processed_channels.append(processed)

            self.processed_image = np.stack(processed_channels, axis=2)

        if hasattr(self, 'image_display'):
            self.image_display.set_data(self.processed_image)
            self.image_display.set_clim(0, 255)

    def update_plot(self):
        """Обновление графика (без пересоздания кнопок)"""

        self.ax.clear()

        if len(self.points) < 2:
            self.ax.set_title('Добавьте хотя бы 2 точки (нажмите "Добавить" и кликните)',
                              fontsize=12, fontweight='bold')
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 255)
            self.ax.grid(True, alpha=0.3)
            self.update_mode_text()
            self.process_and_update_image()
            self.fig.canvas.draw_idle()
            return

        self.interpolation.change_points(self.points)

        self.process_and_update_image()

        points = sorted(self.points, key=lambda p: p[0])
        x_nodes, y_nodes = zip(*points)

        x_smooth = np.linspace(min(x_nodes), max(x_nodes), 500)
        y_smooth = [self.interpolation.interpolate_single(x) for x in x_smooth]

        self.ax.plot(x_smooth, y_smooth, 'b-', linewidth=2.5, label='Интерполяция', alpha=0.8, zorder=1)
        self.ax.plot(x_nodes, y_nodes, 'r--', linewidth=1, alpha=0.5, label='Отрезки')
        self.ax.scatter(x_nodes, y_nodes, color='red', s=120, zorder=5, label='Узлы', picker=True)


        margin_x = max(10, (max(x_nodes) - min(x_nodes)) * 0.1)
        margin_y = max(10, (max(y_nodes) - min(y_nodes)) * 0.1)
        self.ax.set_xlim(min(x_nodes) - margin_x, max(x_nodes) + margin_x)
        self.ax.set_ylim(min(y_nodes) - margin_y, max(y_nodes) + margin_y)

        self.ax.set_xlabel('Входное значение (пиксель)', fontsize=12)
        self.ax.set_ylabel('Выходное значение (пиксель)', fontsize=12)
        self.ax.set_title("Интерактивная линейная интерполяция", fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='upper left', fontsize=10)

        self.mode_text = ''

        self.update_mode_text()
        self.update_text()
        self.update_hist()
        self.fig.canvas.draw_idle()

    def update_mode_text(self):
        """Обновление текста режима"""
        if hasattr(self, 'mode_text'):
            if self.add_mode:
                self.mode_text = 'РЕЖИМ ДОБАВЛЕНИЯ: кликните для добавления точки'
            else:
                self.mode_text = 'РЕЖИМ РЕДАКТИРОВАНИЯ: перетаскивайте точки'

    def on_press(self, event):
        """Обработчик нажатия мыши"""
        if event.inaxes != self.ax:
            return

        if self.add_mode and event.xdata is not None:
            x, y = event.xdata, event.ydata
            x = np.clip(x, 0, 255)
            y = np.clip(y, 0, 255)
            self.points.append((x, y))
            self.update_plot()
            return

        for i, (x, y) in enumerate(self.points):
            if abs(event.xdata - x) < 5 and abs(event.ydata - y) < 5:
                self.selected_point = i
                self.dragging = True
                break

    def on_release(self, event):
        """Обработчик отпускания мыши"""
        self.selected_point = None
        self.dragging = False

    def on_motion(self, event):
        """Обработчик движения мыши"""
        if self.dragging and self.selected_point is not None and event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                x = np.clip(x, 0, 255)
                y = np.clip(y, 0, 255)
                self.points[self.selected_point] = (x, y)
                self.update_plot()

    def toggle_add_mode(self, event):
        """Переключение режима добавления"""
        self.add_mode = not self.add_mode
        self.update_mode_text()
        self.fig.canvas.draw_idle()

    def clear_points(self, event):
        """Очистка всех точек"""
        self.points = []
        self.update_plot()

    def reset_points(self, event):
        """Сброс к начальным точкам"""
        self.points = [(0, 0), (85, 85), (170, 170), (255, 255)]
        self.add_mode = False
        self.update_plot()

    def show(self):
        """Отображение интерфейса"""
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    interactive = InteractiveInterpolation(image_path='pic/redcat.jpg')
    interactive.show()