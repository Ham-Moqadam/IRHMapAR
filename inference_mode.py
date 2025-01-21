


### ---------------------------------------------------------------------------
### author:             H.Moqadam
### date:               11.1.25
### desc:               Inference mode for the autoregressive trained model
### desc:               >> choosing small data and fixing the bugs. <<
### 
### ---------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random


## -- OLD PRWEDICITON WITH THEM 


# Load the saved model
saved_model = load_model('trained_model_more_data.keras')


gram_piece = np.loadtxt("../DATA/grams/20023150_patch_16.csv", delimiter = ",")
mask_piece = np.loadtxt("../DATA/masks/20023150_patch_16.csv", delimiter = ",")

n = random.randrange(0,448)
m = n + 64

gram_piece_small = gram_piece[n:m, n:m]
mask_piece_smal = mask_piece [n:m, n:m]

input_data = gram_piece_small.reshape(1, 64, 64, 1)

predictions = saved_model.predict(input_data)

predicted_irh = predictions[0]
predicted_irh =  normalize_matrix(predicted_irh, 0, 1)
prediction_stopping = predictions[1]
prediction_stopping = normalize_matrix(prediction_stopping , 0, 1)

print(f"the random number for the patch is {n} and second one {m}")


plt.figure(figsize=(10,10))
plt.subplot(221)
plt.title("radargram")
plt.imshow(gram_piece_small)
plt.subplot(222)
plt.title("mask")
plt.imshow(mask_piece_smal)
plt.subplot(223)
plt.title("output - predicted horizon")
plt.imshow(predicted_irh[0,:,:,0])
plt.subplot(224)
plt.title("ouput - prediction stopping")
plt.imshow(prediction_stopping[0,:,:,0])
plt.tight_layout()
# plt.savefig("./res_5_BC.png")


#%% ------------------- NEW PREDICTION (ITERATIVELY) ---------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

















