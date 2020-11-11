import cv2 as cv
import numpy as np


def get_img_info(img: np.ndarray):
    print("type: ", type(img))
    print("shape: ", img.shape)
    print("size: ", img.size)  # total num of pixels
    print("dtype: ", img.dtype)  # uint8
    print(img)


def resize_demo(img: np.ndarray):
    rs1 = cv.resize(img, (600, 1000))  # (x-axis, y-axis)
    cv.imshow("Resize 1", rs1)
    rs2 = cv.resize(img, (0, 0), fx=1.5, fy=1)  # lx = lx * 1.5
    cv.imshow("Resize 2", rs2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def fill_border(img: np.ndarray):
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    border_replicate = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REPLICATE)
    cv.imshow("BORDER_REPLICATE", border_replicate)
    border_reflect = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT)
    cv.imshow("BORDER_REFLECT", border_reflect)
    border_reflect101 = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT_101)
    cv.imshow("BORDER_REFLECT_101", border_reflect101)
    border_wrap = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_WRAP)
    cv.imshow("BORDER_WRAP", border_wrap)
    border_constant = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT)
    cv.imshow("BORDER_CONSTANT", border_constant)


if __name__ == '__main__':

    # read image
    src = cv.imread('./Lenna.png')  # ndarray
    cv.imshow('color image', src)
    get_img_info(src)

    # read gray image
    src_gray = cv.imread('./Lenna.png', flags=cv.IMREAD_GRAYSCALE)
    cv.imshow("gray image", src_gray)

    # flip
    # 0: flip around the y-axis
    # positive value (e.g. 1): flip around the x-axis
    # negative value (e.g. -1): flip around both axes
    src = cv.flip(src, flipCode=0)

    # convert color img to gray scale
    gray_img = cv.cvtColor(src, code=cv.COLOR_BGR2GRAY)

    # save image
    cv.imwrite('./Lenna_save.png', gray_img)

    # resize
    src1 = cv.imread('./girl_with_glasses.jpg')
    resize_demo(src1)

    src2 = cv.imread('./Lenna.png')
    fill_border(src2)

    cv.waitKey(0)
    cv.destroyAllWindows()