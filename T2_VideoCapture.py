import cv2 as cv


def video_demo():
    """
    captured by webcam
    """
    capture = cv.VideoCapture(0)  # 0 will return video from 1st webcam
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, flipCode=1)
        cv.imshow("video", frame)
        wait = cv.waitKey(50)  # -1
        if wait == 27:  # press Esc to exit
            break


def video_demo_2(vid_path):
    capture = cv.VideoCapture(vid_path)
    if capture.isOpened():
        flag_open, frame = capture.read()  # read first frame
        while flag_open:
            cv.imshow("Video", frame)
            if cv.waitKey(10) & 0xFF == 27:  # press Exc
                break
            flag_open, frame = capture.read()  # read next frame

    capture.release()


if __name__ == '__main__':
    # video_demo()
    video_demo_2("track_red.MOV")
    cv.destroyAllWindows()