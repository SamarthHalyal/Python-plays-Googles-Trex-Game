import numpy as np
import models

WIDTH = 30
HEIGHT = 18
LR = 1e-3
EPOCH = 8
MODEL_NAME = 'Trex-{}-{}-{}-epochs.model'.format(LR,'alexnet',EPOCH)

model = models.alexnet(WIDTH,HEIGHT,LR,2)

train_data = np.load('training_data_v2.npy')

train = train_data[:-50]
test = train_data[-50:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

model.fit({'input': X},{'targets': y}, n_epoch=EPOCH,
          validation_set=({'input': test_X}, {'targets': test_y}),
          snapshot_step=50, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)


