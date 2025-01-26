import dlib
import cv2
from matplotlib import pyplot as plt


def mostrar(img):
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.show()


imagem = cv2.imread('Images/people2.jpg')
detector_face = dlib.cnn_face_detection_model_v1(
    'Weights/mmod_human_face_detector.dat')

deteccoes = detector_face(imagem, 1)

for face in deteccoes:
    l, t, r, b, c = face.rect.left(), face.rect.top(
    ), face.rect.right(), face.rect.bottom(), face.confidence
    cv2.rectangle(imagem, (l, t), (r, b), (0, 255, 0), 2)

mostrar(imagem)
