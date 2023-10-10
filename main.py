import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm


def distance(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    return np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))


def main():
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))

    img1 = cv2.imread("ross1.jpg")
    img2 = cv2.imread("ross2.jpg")
    img3 = cv2.imread("chandler1.jpg")
    img4 = cv2.imread("chandler2.jpg")

    [person1] = app.get(img1)
    [person2] = app.get(img2)
    [person3] = app.get(img3)
    [person4] = app.get(img4)

    people = [("Ross 1", person1), ("Ross 2", person2), ("Chandler 1", person3), ("Chandler 2", person4)]
    for i in range(len(people)-1):
        print(f"{people[i][0]} vs {people[i+1][0]}: {distance(people[i][1]['embedding'], people[i+1][1]['embedding'])}")


if __name__ == '__main__':
    main()
