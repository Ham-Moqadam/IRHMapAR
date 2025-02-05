


### ---------------------------------------------------------------------------
### author:             H.Moqadam
### date:               11.1.25
### desc:               Inference mode for the autoregressive trained model
### desc:               >> choosing small data and fixing the bugs. <<
### 
### ---------------------------------------------------------------------------



#%% ------------------- NEW PREDICTION (ITERATIVELY) ---------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model



def normalize_matrix(image, lower_range, upper_range):
    """
    normalizes the image, within the given the upper and lower limit
    """
    import numpy as np

    a = np.array((image - np.min(image)) / (np.max(image) - np.min(image)))
    b = upper_range - lower_range
    c = lower_range
    answer = a*b + c
    return answer



def label_connected_comps(image, size_threshold, connectivity):
    """
    Label the connected components
    Calculate the size of each component
    Define the size threshold
    Create a mask for components larger than the threshold
    Apply the mask to the labeled image to remove small components
    Convert the filtered image back to binary format

    Parameters
    ----------
    image : np.ndarray
        Binary input image.
    size_threshold : int
        Minimum size of components to keep.
    connectivity : int
        Connectivity for labeling (4 or 8).

    Returns
    -------
    filtered_binary_image : np.ndarray
        Binary image with small components removed.
    """
    import numpy as np
    from scipy.ndimage import label

    if connectivity == 4:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
    elif connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    labeled_image, num_labels = label(image, structure=structure)

    component_sizes = np.bincount(labeled_image.ravel())
    large_components_mask = np.isin(labeled_image, np.where(component_sizes >= size_threshold)[0])
    filtered_image = labeled_image * large_components_mask
    filtered_binary_image = (filtered_image > 0).astype(np.uint8)

    return filtered_binary_image



import tensorflow as tf
import keras
@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    y_true_f = tf.keras.backend.flatten(y_true)  # Flatten to a 1D array
    y_pred_f = tf.keras.backend.flatten(y_pred)  # Flatten to a 1D array
    intersection = tf.reduce_sum(y_true_f * y_pred_f)  # Intersection of prediction & truth
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


import keras as K
@keras.saving.register_keras_serializable()
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0) errors
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy)
    return loss



import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

@keras.saving.register_keras_serializable()
def combined_loss(y_true, y_pred):
    bce_loss = BinaryCrossentropy()(y_true, y_pred)
    dice_loss_value = dice_loss(y_true, y_pred)
    return 0.5 * bce_loss + 0.5 * dice_loss_value



    

#%%




# Load the trained model
saved_model = load_model("")



### --- 0. getting the radargram and mask
# Load radargram and mask patches
gram_piece = np.loadtxt("../DATA/grams/20023150_patch_16.csv", delimiter=",")
mask_piece = np.loadtxt("../DATA/masks/20023150_patch_16.csv", delimiter=",")

# Select a random 64x64 patch
n = random.randrange(0, 448)
m = n + 64
gram_piece_small = gram_piece[n:m, n:m]
mask_piece_small = mask_piece[n:m, n:m]


#%%
## ----------------------------------

# Prepare the radargram input
radargram_input = gram_piece_small.reshape(1, 64, 64, 1)


# Prepare the additional input (example: all zeros of the same shape as radargram_input)
additional_input = np.zeros_like(radargram_input)  # Shape: (1, 64, 64, 1)
# for the final answer make a 1d array to be appended to
javab = np.zeros((64,64))


column_no = 0
until_column = 64


for column_no in range(0, until_column):
    # Predict using both inputs
    predictions = saved_model.predict([radargram_input, additional_input])
    
    ## -- PREDICTED IRH
    predicted_irh = predictions[0]
    predicted_irh = normalize_matrix(predicted_irh, 0, 1)[0,:,:,0]
    ## -- cropping and padding

    ## -- STOPPING CRITERIA
    stopping_cr =  predictions[1]
    stopping_cr =  normalize_matrix(stopping_cr, 0, 1)[0,:,:,0]

    ##! the choice of output 
    array_to_work_with = np.copy(predicted_irh)
    
    """
    ## finding where the max is
    max_index = np.argmax(array_to_work_with[:, column_no])

    ## setting the max location's element to 1
    array_to_work_with[max_index, column_no] = 99
    """
        
    # Get the column values
    col_values = array_to_work_with[:, column_no]
    
    # Get the stopping criterion values for the column
    stopping_values = stopping_cr[:, column_no]
    
    # Find the index of the maximum value in the column
    max_index = np.argmax(col_values)
    
    # **Skip updating if stopping_cr at this pixel is > 0.9**
    if stopping_values[max_index] > 0.9:
        print(f"Skipping column {column_no} due to stopping criterion at row {max_index}.")
    else:
            array_to_work_with[max_index, column_no] = 99

    
    # plt.figure("plotting the argmax set to ONE")
    # plt.imshow(array_to_work_with)
    ## setting others to zero
    array_to_work_with[array_to_work_with[:, column_no] < 99, column_no] = 0
    array_to_work_with[array_to_work_with == 99] = 1
    # plt.figure(f"plotting the rest of column {column_no} set to ONE")
    # plt.imshow(array_to_work_with)
    ## seting all the next columns to zero
    if column_no < 63:
        array_to_work_with[:, column_no+1:] = 0
    # plt.figure("plotting the next clumns all set to ZERO")
    # plt.imshow(array_to_work_with)
    # plt.close("all")
    javab[:,column_no] = array_to_work_with[:,column_no]
    # plt.figure(f"plotting the JAVAB which has {column_no} columns")
    # plt.imshow(javab)
    # plt.savefig(f"with_{column_no}_columns.png")
    ## -- adding a dimension
    additional_input = array_to_work_with.reshape((1, 
                                                   array_to_work_with.shape[0], 
                                                   array_to_work_with.shape[1], 
                                                   1))


#%% POST - PROCESSING


from skimage.morphology import skeletonize
import cv2 as cv
kernel = np.ones((3,3))
dilated_javab = cv.dilate(javab,kernel,iterations = 1)
# plt.imshow(dilated_javab)
# eroded_javab = cv.erode(dilated_javab,kernel,iterations = 1)
# plt.imshow(eroded_javab)
skeleton_dilated = skeletonize(dilated_javab)
# plt.imshow(skeleton_dilated)
cca_sk_dil = label_connected_comps(skeleton_dilated, 5, 8)



plt.figure("inferred and postprocessed", figsize=(14, 10))
plt.subplot(231)
plt.title("Radargram")
plt.imshow(gram_piece_small, cmap="viridis")
plt.subplot(232)
plt.title("Output - Predicted Horizon")
plt.imshow(predicted_irh)
plt.subplot(233)
plt.title("Output - stopping criteria")
plt.imshow(stopping_cr)
plt.subplot(234)
plt.title("Label")
plt.imshow(mask_piece_small, cmap="viridis")
plt.subplot(235)
plt.title("output")
plt.imshow(javab)
plt.subplot(236)
plt.title("output - post")
plt.imshow(cca_sk_dil)
plt.tight_layout()
# plt.savefig("2input_weighted_NL.png")



# plt.close("all")
## appending it to an emprty mask
# if column_no == 0:
#     javab[:,0] = array_to_work_with[:,column_no]
# else:
#     javab = np.column_stack((javab, array_to_work_with[:, column_no]))



#%% WITH 5 PIXEL LIMITATION


## Prepare the radargram input
radargram_input = gram_piece_small.reshape(1, 64, 64, 1)

## Prepare the additional input (example: all zeros of the same shape as radargram_input)
additional_input = np.zeros_like(radargram_input)  # Shape: (1, 64, 64, 1)
## for the final answer make a 1d array to be appended to
#javab = np.zeros((64,64))

until_column = 64


column_no = 0


javab = np.empty((64, 1))  # If you want to keep 1 initial column


for column_no in range(0, until_column):
    ## 1. Predict using both inputs
    predictions = saved_model.predict([radargram_input, additional_input])

    ## 2. selecting the array to work with (for now predicted_IRH) + changing shape
    ## TODO: working with the stopping_cr as well again!
    pred_irh = predictions[0][0,:,:,0]   
    print(f"shape is {pred_irh.shape} | min is {np.min(pred_irh)} | max is {np.max(pred_irh)}")
    ## 3. normalizing it
    pred_irh_norm = normalize_matrix(pred_irh, 0, 1)
    working_array = np.copy(pred_irh_norm)
    # plt.hist(working_array )
    # plt.plot(working_array [:,0])
    # plt.subplot(121), plt.imshow(gram_piece_small), plt.subplot(122), plt.imshow(working_array)
    ## 4. find the index of the max of the "column_no" column
    col_values = working_array[:, column_no]
    max_index = np.argmax(col_values)
    plt.imshow(working_array)
    working_array[max_index, column_no] = 99.0
    working_array[working_array<99.0] = 0
    working_array[working_array ==99.0] = 1.0

    plt.close()
    
    if column_no == 0:
        javab[:,0] = working_array[:,0]
    else:
        javab = np.column_stack((javab, working_array[:,column_no]))

    plt.figure("new loop", figsize=(15,11)), plt.suptitle(f"plot for iteration number {column_no +1} ", fontweight="bold", fontsize=14)
    plt.subplot(231), plt.imshow(radargram_input[0,:,:,0]), plt.title("gram, 1st input")
    plt.subplot(232), plt.imshow(additional_input[0,:,:,0]), plt.title("$2^{nd}$ input for inference")
    plt.subplot(233), plt.imshow(pred_irh_norm), plt.title("this loops inference")
    plt.subplot(234), plt.imshow(working_array), plt.title("result of this loop (this column)")

    ##DITO: the problem is that the input every time is onlty one pixel as 1 in the 
    ## entire array! 
    additional_input[0,:,:,0] =  additional_input[0,:,:,0] + working_array
    additional_input[additional_input >1] = 1
    ## add dimension to be able to predict using it

    plt.subplot(235), plt.imshow(additional_input[0,:,:,0]) , plt.title("going as input for next loop")
    plt.subplot(236), plt.imshow(javab), plt.title("result"), plt.tight_layout()






















































