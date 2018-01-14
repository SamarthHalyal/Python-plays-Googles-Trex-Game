import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W

def draw_lines(img,lines):
    flag = False
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), (255,255,255), 1)
            if coords[1] < 133 or coords[3] < 133:
                flag = True
    except:
        pass
    return flag

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, (255,255,255))
    masked = cv2.bitwise_and(img, mask)
    return masked

def main():
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,85,300,265)))

        processed_img = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
        processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
        vertices = np.array([[100,0],[150,0],[150,265],[100,265]], np.int32)
        processed_img = roi(processed_img, [vertices])

        lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, -180,np.array([]),0,50)
        flag = draw_lines(processed_img,lines)
        if flag:
            PressKey(W)
            PressKey(W)
            ReleaseKey(W)
        print("time passed per in seconds : ",time.time()-last_time)
        last_time = time.time()
        cv2.imshow('window', processed_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break     
        


main()
