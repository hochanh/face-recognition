import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm


def cosine_distance(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    return np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))


def main():
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))

    img1 = cv2.imread("ross1.jpg")
    img2 = cv2.imread("ross2.jpg")
    img3 = cv2.imread("chandler1.jpg")

    [person1] = app.get(img1)
    [person2] = app.get(img2)
    [person3] = app.get(img3)

    print(f"1 vs 2: {cosine_distance(person1['embedding'], person2['embedding'])}")
    print(f"2 vs 3: {cosine_distance(person1['embedding'], person3['embedding'])}")


if __name__ == '__main__':
    main()
