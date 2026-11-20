import cv2

def capture_video():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Открываем первый подключенный источник видео (обычно это веб-камера)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return

    while True:
        # Захват кадра
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: Не удалось захватить кадр.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(1, 1))
        
        for (x, y, w, h) in faces:
            # Извлечение области лица
            face_region = frame[y:y+h, x:x+w]

            # Применение размытия
            frame_ = cv2.GaussianBlur(face_region, (23, 23), 100)

            # Замена области лица размытым изображением
            frame[y:y+h, x:x+w] = frame_
            # Отображение кадра
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Видео с камеры', frame)

        # Выход при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Запуск функции захвата видео
capture_video()