import cv2
from matplotlib import pyplot as plt
imagem = cv2.imread('Images/people1.jpg')


def mostrar(img):
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.show()


imagem = cv2.resize(imagem, (800, 600))
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detector_facial = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
# após vários teste, essa é a acurácia ideal para essa imagem
deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.485)

for x, y, w, h in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 5)

mostrar(imagem)
