from PIL import Image
import cv2
import numpy as np
import os

# preparação as imagens para o treinamento


def get_image_data():
    paths = [os.path.join('yalefaces\\train', f)
             for f in os.listdir('yalefaces\\train')]
    # print(paths)
    faces = []
    ids = []
    for path in paths:
        # print(path)
        imagem = Image.open(path).convert('L')
        imagem_np = np.array(imagem, 'uint8')
        # print(type(imagem_np))
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(imagem_np)

    return np.array(ids), faces


ids, faces = get_image_data()

# treinamento do classificador LBPH
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('2- Reconhecimento facial//lbph_classifier.yml')
