import os
import pathlib

import cv2
import face_recognition


def main():
    data_root_orig = './testData'
    data_root = pathlib.Path(data_root_orig)

    face_root = './testData-face'

    all_image_paths = list(data_root.glob('*/*.jpg'))
    count = 0

    for fi in all_image_paths:
        ps = (fi.parts[1], fi.parts[2])
        print(ps)
        print(fi)
        image = cv2.imread(str(fi))
        fls = face_recognition.face_locations(image)

        for f in fls:
            print(f)
            newFile = os.path.join(face_root, fi.parts[1], 'face-' + str(count) + '-' + fi.parts[2])
            print('newFile:', newFile)
            cv2.imwrite(newFile, image[f[0]:f[2], f[3]:f[1]])
            count += 1


if __name__ == '__main__':
    main()
