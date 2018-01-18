import pandas as pd
import numpy as np
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data_v2.npy')
##print(len(train_data))
df = pd.DataFrame(train_data)
ws = []
Nones = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    keys = data[1]

    if keys == [1,0]:
        ws.append([img,keys])
    elif keys == [0,1]:
        Nones.append([img,keys])
    else:
        print("No matches")
        
ws = ws[:len(Nones)]
Nones = Nones[:len(ws)]

final_data = ws + Nones

shuffle(final_data)
print(len(final_data)) 
np.save('training_data_v2.npy',final_data)

print(Counter(df[1].apply(str)))

##for data in train_data:
##    img = data[0]
##    keys = data[1]
##    cv2.imshow('test',img)
##    print(keys)
##    if cv2.waitKey(25) & 0XFF == ord('q'):
##        cv2.destroyAllWindows()
##        break
