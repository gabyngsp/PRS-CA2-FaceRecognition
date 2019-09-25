import os

import cv2
import face_recognition


def videoToImages(className, filename):
    print('Converting ', filename)
    vidcap = cv2.VideoCapture(os.path.join('rawData', className, filename))
    success, image = vidcap.read()
    count = 0
    while success:
        # rows, cols, channels = image.shape

        if className in ('XiaoYan', 'GabyNg'):
            image = cv2.transpose(image)
        elif className in ('LeeSeng'):
            image = cv2.transpose(image)
            image = cv2.flip(image, 0)


        ## with face clipping
        fls = face_recognition.face_locations(image)
        for f in fls:

            cv2.imwrite(os.path.join('data-face', className, filename) + '-%d.jpg' % count, image[f[0]:f[2], f[3]:f[1]])
            count += 1

        # without face clipping
        cv2.imwrite(os.path.join('data-full', className, filename) + '-%d.jpg' % count, image)
        count += 1

        success, image = vidcap.read()
        # if success:
        #     print('.', end='')
        # else:
        #     print('X', end='')

        # print('Read new frame:', success)



def main():
    if not os.path.exists(os.path.join('data-full', 'LeeSeng')):
        print('Creates folder.')
        os.makedirs(os.path.join('data-full', 'LeeSeng'))
        os.makedirs(os.path.join('data-full', 'XiaoYan'))
        os.makedirs(os.path.join('data-full', 'GabyNg'))

    if not os.path.exists(os.path.join('data-face', 'LeeSeng')):
        print('Creates folder.')
        os.makedirs(os.path.join('data-face', 'LeeSeng'))
        os.makedirs(os.path.join('data-face', 'XiaoYan'))
        os.makedirs(os.path.join('data-face', 'GabyNg'))

    print('Converting Videos to images...')
    videoToImages('LeeSeng', 'VID_20190902_091449.mp4')
    videoToImages('LeeSeng', 'VID_20190905_204615.mp4')
    videoToImages('LeeSeng', 'VID_20190908_145635.mp4')

    videoToImages('XiaoYan', 'IMG_4749.MOV')
    videoToImages('XiaoYan', 'IMG_4747.MOV')
    videoToImages('XiaoYan', 'IMG_4705.MOV')

    videoToImages('GabyNg', '20190902_124535.mp4')
    videoToImages('GabyNg', '20190902_124712.mp4')
    print('== END ==')


def main_face_test():
    image = face_recognition.load_image_file(os.path.join('data', 'LeeSeng', 'VID_20190902_091449.mp4-0.jpg'))
    face_locations = face_recognition.face_locations(image)
    f = face_locations[0]
    print(face_locations, ' ', f)
    a = image[f[0]:f[2], f[3]:f[1]]
    print('a ', a.shape)


if __name__ == '__main__':
    main()
