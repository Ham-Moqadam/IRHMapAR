


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
    

#%%




# Load the trained model
saved_model = load_model('trained_model_more_data_emptymask_2input.keras')



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
    """
    predicted_irh = predicted_irh[1:-1,1:-1]
    predicted_irh = np.pad(predicted_irh, 
                           pad_width=1, 
                           mode='symmetric')
    predicted_irh = normalize_matrix(predicted_irh, 0, 1)
    """
    
    ## -- STOPPING CRITERIA
    stopping_cr =  predictions[1]
    stopping_cr =  normalize_matrix(stopping_cr, 0, 1)[0,:,:,0]
    ## -- cropping and padding
    """
    stopping_cr  = stopping_cr [1:-1,1:-1]
    stopping_cr = np.pad(stopping_cr , 
                           pad_width=1, 
                           mode='symmetric')
    stopping_cr =  normalize_matrix(stopping_cr, 0, 1)
    """
    
    # Visualization
    # plt.figure(figsize=(10, 10))
    # plt.subplot(221)
    # plt.title("Radargram")
    # plt.imshow(gram_piece_small, cmap="gray")
    # plt.subplot(222)
    # plt.title("Label")
    # plt.imshow(mask_piece_small, cmap="gray")
    # plt.subplot(223)
    # plt.title("Output - Predicted Horizon")
    # plt.imshow(predicted_irh)
    # plt.subplot(224)
    # plt.title("Output - stopping criteria")
    # plt.imshow(stopping_cr)
    # plt.tight_layout()
    # plt.show()
    
    ## ! 
    array_to_work_with = np.copy(predicted_irh)
    
    ## finding where the max is
    np.argmax(array_to_work_with[:,column_no])
    ## setting the max location's element to 1
    array_to_work_with[np.argmax(array_to_work_with[:,column_no]),column_no] = 1 
    # plt.figure("plotting the argmax set to ONE")
    # plt.imshow(array_to_work_with)
    ## setting others to zero
    array_to_work_with[array_to_work_with[:, column_no] < 1, column_no] = 0
    # plt.figure(f"plotting the rest of column {column_no} set to ONE")
    # plt.imshow(array_to_work_with)
    ## seting all the next columns to zero
    array_to_work_with[:,column_no+1:] = 0
    # plt.figure("plotting the next clumns all set to ZERO")
    # plt.imshow(array_to_work_with)
    plt.close("all")
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
plt.imshow(dilated_javab)
# eroded_javab = cv.erode(dilated_javab,kernel,iterations = 1)
# plt.imshow(eroded_javab)
skeleton_dilated = skeletonize(dilated_javab)
plt.imshow(skeleton_dilated)
cca_sk_dil = label_connected_comps(skeleton_dilated, 5, 8)



plt.figure(figsize=(14, 10))
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





# plt.close("all")
## appending it to an emprty mask
# if column_no == 0:
#     javab[:,0] = array_to_work_with[:,column_no]
# else:
#     javab = np.column_stack((javab, array_to_work_with[:, column_no]))









































