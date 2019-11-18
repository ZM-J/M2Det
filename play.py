from kfbReader import kfbReader
import cv2 as cv

if __name__ == "__main__":
    data_dir = u'data/raw_data/neg_4/T2019_97.kfb'
    read = kfbReader.reader()
    read.ReadInfo(data_dir, 20, False)
    # a = read.ReadPreview()
    roi = read.ReadRoi(i * 8192, j * 8192, 8192, 8192, 20)
    b = cv.resize(roi, (512, 512))
    cv.imshow('roi', b)
    cv.waitKey(0)
    print(f'Height: {read.getHeight()}')
    print(f'Width: {read.getWidth()}')