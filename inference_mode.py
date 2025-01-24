


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
saved_model = load_model('trained_model_more_data_emptymask_2input.keras')


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

from new_model_sequential import normalize_matrix

### --- 0. getting the radargram and mask
# Load radargram and mask patches
gram_piece = np.loadtxt("../DATA/grams/20023150_patch_16.csv", delimiter=",")
mask_piece = np.loadtxt("../DATA/masks/20023150_patch_16.csv", delimiter=",")

# Select a random 64x64 patch
n = random.randrange(0, 448)
m = n + 64
gram_piece_small = gram_piece[n:m, n:m]
mask_piece_small = mask_piece[n:m, n:m]

# Prepare the radargram input
radargram_input = gram_piece_small.reshape(1, 64, 64, 1)



# Load the trained model
saved_model = load_model('trained_model_more_data_emptymask_2input.keras')


# Prepare the additional input (example: all zeros of the same shape as radargram_input)
additional_input = np.zeros_like(radargram_input)  # Shape: (1, 64, 64, 1)
# for the final answer make a 1d array to be appended to
javab = np.zeros((64,1))


column_no = 63
    
# for i in range(0, 20):
# Predict using both inputs
predictions = saved_model.predict([radargram_input, additional_input])
predicted_irh = predictions[0]
predicted_irh = normalize_matrix(predicted_irh, 0, 1)[0,:,:,0]
## -- making the first and last row of the first column as means of the column
predicted_irh[0,column_no] = np.mean(predicted_irh[:,column_no])
predicted_irh[-1,column_no] = np.mean(predicted_irh[:,column_no])

# Visualization
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title("Radargram")
plt.imshow(gram_piece_small, cmap="gray")
plt.subplot(222)
plt.title("other input")
plt.imshow(additional_input[0,:,:,0], cmap="gray")
plt.subplot(223)
plt.title("Output - Predicted Horizon")
plt.imshow(predicted_irh, cmap="gray")
plt.subplot(224)
plt.title("Label")
plt.imshow(mask_piece_small, cmap="gray")
plt.tight_layout()
plt.show()

## finding where the max is
np.argmax(predicted_irh[:,column_no])
## setting the max location's element to 1
predicted_irh[np.argmax(predicted_irh[:,column_no]),column_no] = 1 
plt.figure("plotting the argmax set to ONE")
plt.imshow(predicted_irh)
## setting others to zero
predicted_irh[predicted_irh[:, column_no] < 1, column_no] = 0
plt.figure(f"plotting the rest of column {column_no} set to ONE")
plt.imshow(predicted_irh)
## seting all the next columns to zero
predicted_irh[:,column_no+1:] = 0
plt.figure("plotting the next clumns all set to ZERO")
plt.imshow(predicted_irh)
# plt.close("all")
## appending it to an emprty mask
javab = np.column_stack((javab, predicted_irh[:, column_no]))
plt.figure(f"plotting the JAVAB which has {column_no} columns")
# plt.close("all")
plt.imshow(javab)
plt.savefig(f"with_{column_no}_columns.png")
## -- adding a dimension
additional_input = predicted_irh.reshape((1, predicted_irh.shape[0], predicted_irh.shape[1], 1))
# 



















































