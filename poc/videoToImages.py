import cv2

def leesengVideoToImages():
    vidcap = cv2.VideoCapture('rawData/LeeSeng/VID_20190902_091449.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        # rows, cols, channels = image.shape
        # M =cv2.getRotationMatrix2D((cols/2, rows/2),90,1)
        # nImage = cv2.warpAffine( image, M, (rows, cols))
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)

        cv2.imwrite('data/LeeSeng/frame%d.jpg' % count, image)
        success, image = vidcap.read()
        if success:
            print('.', end='')
        else:
            print('X', end='')

        # print('Read new frame:', success)
        count+=1

def gabyNgVideoToImages():
    vidcap = cv2.VideoCapture('rawData/GabyNg/20190902_124712.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        # rows, cols, channels = image.shape
        # M =cv2.getRotationMatrix2D((cols/2, rows/2),90,1)
        # nImage = cv2.warpAffine( image, M, (rows, cols))
        image = cv2.transpose(image)
        # image = cv2.flip(image, 0)

        cv2.imwrite('data/GabyNg/bframe%d.jpg' % count, image)
        success, image = vidcap.read()
        if success:
            print('.', end='')
        else:
            print('X', end='')

        # print('Read new frame:', success)
        count+=1

def gabyNgVideoToImages():
    vidcap = cv2.VideoCapture('rawData/GabyNg/20190902_124712.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        # rows, cols, channels = image.shape
        # M =cv2.getRotationMatrix2D((cols/2, rows/2),90,1)
        # nImage = cv2.warpAffine( image, M, (rows, cols))
        image = cv2.transpose(image)
        # image = cv2.flip(image, 0)

        cv2.imwrite('data/GabyNg/bframe%d.jpg' % count, image)
        success, image = vidcap.read()
        if success:
            print('.', end='')
        else:
            print('X', end='')

        # print('Read new frame:', success)
        count+=1

def xiaoYanVideoToImages():
    vidcap = cv2.VideoCapture('rawData/XiaoYan/IMG_4705.MOV')
    success, image = vidcap.read()
    count = 0
    while success:
        # rows, cols, channels = image.shape
        # M =cv2.getRotationMatrix2D((cols/2, rows/2),90,1)
        # nImage = cv2.warpAffine( image, M, (rows, cols))
        image = cv2.transpose(image)
        # image = cv2.flip(image, 0)

        cv2.imwrite('data/XiaoYan/xframe%d.jpg' % count, image)
        success, image = vidcap.read()
        if success:
            print('.', end='')
        else:
            print('X', end='')

        # print('Read new frame:', success)
        count+=1

def main():
    # leesengVideoToImages()
    gabyNgVideoToImages()
    xiaoYanVideoToImages()


if __name__ == '__main__':
    main()