import cv2
from matplotlib import pyplot as plt
imagem = cv2.imread('Images/people1.jpg')


def mostrar(img):
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.show()


imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detector_olhos = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
deteccoes_olhos = detector_olhos.detectMultiScale(
    imagem_cinza, scaleFactor=1.09, minNeighbors=10, maxSize=(70, 70))
for (x, y, w, h) in deteccoes_olhos:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 0, 255), 3)


detector_facial = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
detectocao_facial = detector_facial.detectMultiScale(
    imagem_cinza, scaleFactor=1.485)
for (x, y, w, h) in detectocao_facial:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 3)

mostrar(imagem)
deteccoes_olhos
