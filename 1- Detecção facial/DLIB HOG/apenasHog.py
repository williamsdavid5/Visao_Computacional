import dlib
import cv2
from matplotlib import pyplot as plt

imagem = cv2.imread('Images/people2.jpg')


def mostrar(img):
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.show()


detector_hog = dlib.get_frontal_face_detector()
deteccoes = detector_hog(imagem, 1)

for face in deteccoes:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(imagem, (l, t), (r, b), (0, 255, 0), 2)

mostrar(imagem)
