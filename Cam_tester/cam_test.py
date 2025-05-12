import cv2

for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(i)
        cv2.imshow('frame', cap.read()[1])
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
