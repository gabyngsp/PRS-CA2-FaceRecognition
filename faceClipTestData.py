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
        
        if face_locations == []:
            # get image height, width
            (h, w) = image.shape[:2]
            # calculate the center of the image
            center = (w / 2, h / 2)
            angle=[90,180,270]
            scale = 1.0
            for idx in range(len(angle)):
                tmp = cv2.getRotationMatrix2D(center, angle[idx], scale)
                rotated = cv2.warpAffine(image, tmp, (h, w))
                face_locations = face_recognition.face_locations(rotated)
                if face_locations == []:
                    continue
                else:
                    for f in face_locations:
                        print(f)
                        newFile = os.path.join(face_root, fi.parts[1], 'face-' + str(count) + '-' + fi.parts[2])
                        print('newFile:', newFile)
                        cv2.imwrite(newFile, image[f[0]:f[2], f[3]:f[1]])
                        count += 1
                    exit
        else:
            for f in fls:
                print(f)
                newFile = os.path.join(face_root, fi.parts[1], 'face-' + str(count) + '-' + fi.parts[2])
                print('newFile:', newFile)
                cv2.imwrite(newFile, image[f[0]:f[2], f[3]:f[1]])
                count += 1


if __name__ == '__main__':
    main()
