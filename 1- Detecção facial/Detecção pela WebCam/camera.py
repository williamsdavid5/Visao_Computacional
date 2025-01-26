import cv2

detector = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(0)

while True:
    ok, frame = videoCapture.read()

    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    deteccoes = detector.detectMultiScale(imagem_cinza)

    for (x, y, w, h) in deteccoes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
