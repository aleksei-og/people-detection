"""
Программа для детекции людей на видео с использованием YOLOv8.
Автоматически загружает предобученные веса, обрабатывает видео и сохраняет результат.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys


class PeopleDetector:
    """
    Класс для детекции людей на видео с использованием YOLOv8.

    Attributes:
        model: Загруженная модель YOLOv8
        confidence_threshold: Порог уверенности для детекции
        class_id_person: ID класса 'person' в модели YOLO
    """

    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Инициализация детектора.

        Args:
            model_path: Путь к файлу весов модели YOLO
            confidence_threshold: Порог уверенности для отображения детекций
        """
        self.confidence_threshold = confidence_threshold
        self.class_id_person = 0  # ID класса 'person' в YOLO

        print("Загрузка модели YOLOv8...")
        try:
            self.model = YOLO(model_path)
            print("Модель успешно загружена")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Пытаюсь скачать предобученные веса...")
            self.model = YOLO('yolov8n.pt')

    def process_frame(self, frame):
        """
        Обработка одного кадра видео.

        Args:
            frame: Входной кадр видео

        Returns:
            Кадр с нарисованными bounding boxes
        """
        # Выполнение детекции
        results = self.model(frame, verbose=False)

        # Отрисовка результатов
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Проверяем, что это человек и уверенность выше порога
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == self.class_id_person and conf >= self.confidence_threshold:
                        # Получаем координаты bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Рисуем прямоугольник
                        color = (0, 255, 0)  # Зеленый цвет
                        thickness = 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                        # Добавляем текст с классом и уверенностью
                        label = f"person: {conf:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2

                        # Вычисляем размер текста для фона
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, text_thickness
                        )

                        # Рисуем фон для текста
                        cv2.rectangle(
                            frame,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            color,
                            -1  # Заполненный прямоугольник
                        )

                        # Рисуем текст
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 5),
                            font,
                            font_scale,
                            (0, 0, 0),  # Черный цвет текста
                            text_thickness
                        )

        return frame

    def process_video(self, input_path, output_path=None):
        """
        Обработка всего видеофайла.

        Args:
            input_path: Путь к входному видеофайлу
            output_path: Путь для сохранения результата (если None, генерируется автоматически)

        Returns:
            Путь к сохраненному видеофайлу
        """
        # Проверяем существование входного файла
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"Ошибка: файл {input_path} не найден")
            return None

        # Генерируем имя выходного файла, если не указано
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"

        print(f"Открытие видеофайла: {input_path}")

        # Открываем видеофайл
        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            print("Ошибка: не удалось открыть видеофайл")
            return None

        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Параметры видео:")
        print(f"  Размер: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Всего кадров: {total_frames}")

        # Создаем VideoWriter для сохранения результата
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print("Начинаю обработку видео...")

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Обрабатываем кадр
            processed_frame = self.process_frame(frame)

            # Записываем обработанный кадр
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 30 == 0:  # Выводим прогресс каждые 30 кадров
                progress = (frame_count / total_frames) * 100
                print(f"Обработано кадров: {frame_count}/{total_frames} ({progress:.1f}%)")

        # Закрываем все ресурсы
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Обработка завершена!")
        print(f"Результат сохранен в: {output_path}")

        return output_path


def main():
    """
    Основная функция программы.
    """
    print("=" * 60)
    print("Детекция людей на видео")
    print("=" * 60)

    # Путь к входному видео
    input_video = "crowd.mp4"

    # Проверяем наличие входного файла
    if not Path(input_video).exists():
        print(f"Ошибка: файл {input_video} не найден в текущей директории")
        print(f"Текущая директория: {Path.cwd()}")
        print("Пожалуйста, поместите файл crowd.mp4 в папку с программой")
        return 1

    # Создаем детектор
    detector = PeopleDetector(confidence_threshold=0.5)

    try:
        # Обрабатываем видео
        output_path = detector.process_video(input_video)

        if output_path:
            print("\n" + "=" * 60)

            print("=" * 60)


        return 0

    except KeyboardInterrupt:
        print("\nОбработка прервана пользователем")
        return 1
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())