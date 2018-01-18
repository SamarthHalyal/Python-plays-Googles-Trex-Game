import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W
from getkeys import key_check
import os

def keys_to_onehot(keys):
    #[W,S]
    output = [0,0]
    if 'W' in keys:
        output[0] = 1
    else:
        output[1] = 1
        
    return output

file_name = 'training.npy'

if os.path.isfile(file_name):
    print("File exits, loading previous data!")
    training_data = list(np.load(file_name))    
else:
    print("File does not exit, starting fresh")
    training_data = []

def main():
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,85,300,265)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(30,18))

        keys = key_check()
        output = keys_to_onehot(keys)
        training_data.append([screen,output])
        
        print("Frame took {} seconds".format(time.time()-last_time))
        last_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)
        
        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break     
        


main()
