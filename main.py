import os
import cv2
import face_recognition


def main():
    imageFiles = os.listdir("./images")
    images_path = []
    encodings = []

    for file in imageFiles:
        images_path.append(f"./images/{file}")

    print("Files:", images_path)

    for image in images_path:
        img = cv2.imread(image)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_econding = face_recognition.face_encodings(rgb_img)[0]
        encodings.append(img_econding)

    img = cv2.imread("unknown.jpg")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_econding = face_recognition.face_encodings(rgb_img)[0]

    result = face_recognition.compare_faces(encodings, img_econding)
    print("Are they the same person?", result)


if __name__ == "__main__":
    main()
