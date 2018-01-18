import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W, S
from getkeys import key_check
import os,models

WIDTH = 30
HEIGHT = 18
LR = 1e-3
EPOCH = 8
MODEL_NAME = 'Trex-{}-{}-{}-epochs.model'.format(LR,'alexnet',EPOCH)

model = models.alexnet(WIDTH,HEIGHT,LR,2)
model.load(MODEL_NAME)

def jump():
    ReleaseKey(S)
    PressKey(W)

def duck():
    ReleaseKey(W)
    PressKey(S)


def main():
    last_time = time.time()
    paused = False
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(0,85,300,265)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(WIDTH,HEIGHT))        
        
        #print("Frame took {} seconds".format(time.time()-last_time))  
        last_time = time.time()

        prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
        moves = list(np.around(prediction))
        print(prediction)

        if moves == [1,0]:
            jump()
            print("Hey\nI\nJumped")
        if moves == [0,1]:
            duck() 
        
        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()
