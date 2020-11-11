import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def dft_demo(img: np.ndarray):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.float32(img)
    dft = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)  # (512, 512, 2)
    dft_shift = np.fft.fftshift(dft)
    # np.log: e based log
    # cv.magnitude: get responding magnitude, input must be float32
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Input image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    return dft_shift


def low_pass_filter(dft: np.ndarray):
    rows, cols, _ = dft.shape
    c_row, c_col = int(rows/2), int(cols/2)
    # generate mask
    mask = np.zeros([rows, cols], dtype=np.uint8)
    mask[c_row-30:c_row+30, c_col-30:c_col+30] = 1

    # IDFT
    lp_dft = cv.bitwise_and(dft, dft, mask=mask)
    lp_dft_shiftback = np.fft.ifftshift(lp_dft)  # shift back to left upper corner
    img_back = cv.idft(lp_dft_shiftback)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121)
    plt.imshow(img_back, cmap='gray')
    plt.title('LowPass Back')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def high_pass_filter(dft: np.ndarray):
    rows, cols, _ = dft.shape
    c_row, c_col = int(rows / 2), int(cols / 2)

    # generate mask
    mask = np.ones([rows, cols], dtype=np.uint8)
    mask[c_row-30:c_row+30, c_col-30:c_col+30] = 0

    # IDFT
    hp_dft = cv.bitwise_and(dft, dft, mask=mask)
    hp_dft_shiftback = np.fft.ifftshift(hp_dft)
    img_back = cv.idft(hp_dft_shiftback)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    plt.subplot(121)
    plt.imshow(img_back, cmap='gray')
    plt.title('HighPass Back')
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    src = cv.imread("./Lenna.png")
    res_dft = dft_demo(src)
    low_pass_filter(res_dft)
    high_pass_filter(res_dft)