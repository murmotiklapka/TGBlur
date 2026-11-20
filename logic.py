import cv2


def process_image(image_path, output_path):
    # Загрузка предобученного классификатора для детектирования лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого для улучшения производительности детектора
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц в изображении
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    # Размытие области вокруг каждого обнаруженного лица
    for (x, y, w, h) in faces:
        # Извлечение области лица
        face_region = image[y:y+h, x:x+w]

        # Применение размытия
        blurred_face = cv2.GaussianBlur(face_region, (23, 23), 100)

        # Замена области лица размытым изображением
        image[y:y+h, x:x+w] = blurred_face

    # Сохранение обработанного изображения
    cv2.imwrite(output_path, image)
    cv2.imshow(output_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Проверяем работу кода
process_image("images.jpg", "output.png")
