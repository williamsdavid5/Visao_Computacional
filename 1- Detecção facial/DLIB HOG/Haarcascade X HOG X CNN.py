import dlib
import cv2
from matplotlib import pyplot as plt


def mostrar(img, titulo):
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.title(titulo)
    plt.show()


def add_text(img, text, pos=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, size=1, color=(0, 255, 0), thickness=2):
    cv2.putText(img, text, pos, font, size, color, thickness, cv2.LINE_AA)


imagem = cv2.imread('Images/people3.jpg')

# HaarCascade
##################################################################################
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

haarCascade = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
# melhor preset possivel para essa imagem
deteccoes = haarCascade.detectMultiScale(imagem_cinza, scaleFactor=1.01)

imagem_haarCascade = imagem.copy()
for (x, y, w, h) in deteccoes:
    cv2.rectangle(imagem_haarCascade, (x, y), (x+w, y+h), (0, 255, 0), 5)
add_text(imagem_haarCascade, "HaarCascade")
print("Cascade OK")

# HOG
##################################################################################
hog = dlib.get_frontal_face_detector()
# melhor preset para evitar um processamento muito lento
deteccoes = hog(imagem_cinza, 3)

imagem_hog = imagem.copy()
for face in deteccoes:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(imagem_hog, (l, t), (r, b), (0, 255, 0), 2)
add_text(imagem_hog, "HOG")
print("HOG OK")

# CNN
##################################################################################
cnn = dlib.cnn_face_detection_model_v1('Weights/mmod_human_face_detector.dat')
deteccoes = cnn(imagem_cinza, 3)

imagem_cnn = imagem.copy()
for face in deteccoes:
    l, t, r, b, c = face.rect.left(), face.rect.top(
    ), face.rect.right(), face.rect.bottom(), face.confidence
    cv2.rectangle(imagem_cnn, (l, t), (r, b), (0, 255, 0), 2)
add_text(imagem_cnn, "CNN")
print("CNN OK")

# mostrar(imagem_haarCascade, "Cascade")
cv2.imshow("Haarcascade", imagem_haarCascade)
cv2.imshow("HOG", imagem_hog)
cv2.imshow("CNN", imagem_cnn)

cv2.waitKey(0)
cv2.destroyAllWindows()
