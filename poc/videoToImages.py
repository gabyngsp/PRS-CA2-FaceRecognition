import os

import cv2


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

        cv2.imwrite(os.path.join('data', className, filename) + '-%d.jpg' % count, image)
        success, image = vidcap.read()
        if success:
            print('.', end='')
        else:
            print('X', end='')

        # print('Read new frame:', success)
        count += 1


def main():
    print('Creates folder.')
    os.makedirs(os.path.join('data', 'LeeSeng'))
    os.makedirs(os.path.join('data', 'XiaoYan'))
    os.makedirs(os.path.join('data', 'GabyNg'))

    print('Converting Videos to images...')
    # leesengVideoToImages()
    # leeseng2VideoToImages()
    videoToImages('LeeSeng', 'VID_20190902_091449.mp4')
    videoToImages('LeeSeng', 'VID_20190905_204615.mp4')

    # xiaoYanVideoToImages()
    videoToImages('XiaoYan', 'IMG_4747.MOV')
    videoToImages('XiaoYan', 'IMG_4705.MOV')

    # gabyNgVideoToImages()
    # gabyNg2VideoToImages()
    videoToImages('GabyNg', '20190902_124535.mp4')
    # videoToImages('GabyNg', '20190902_124712.mp4')
    print('== END ==')


if __name__ == '__main__':
    main()
