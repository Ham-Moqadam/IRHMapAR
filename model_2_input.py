

### ---------------------------------------------------------------------------
### author:             H.Moqadam
### date:               21.1.25
### desc:               The double input model and dataset maker
### desc:               >> choosing small data and fixing the bugs. <<
### 
### ---------------------------------------------------------------------------




## -- general libraries
import os
import time
import numpy as np
import matplotlib.pyplot as plt



## ---- Traversal Logic

import os
import numpy as np
import numpy as np
from skimage.measure import label
from scipy.spatial import cKDTree


def load_mask(filepath):
    """Load a binary mask from a CSV file and convert to integers."""
    mask = np.loadtxt(filepath, delimiter=',', dtype=float)  # Load as float
    return mask.astype(int)  # Explicitly convert to integer



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




def separate_irhs(mask):
    """
    Separate individual IRHs using connected component labeling.
    Each IRH will have a unique integer label.
    """
    labeled_mask = label(mask, connectivity=2)  # 8-connectivity
    return labeled_mask



def traverse_irh(leftmost, irh_coords):
    """
    Traverse IRH pixels starting from the leftmost pixel and progressing 
    through connected neighbors.
    """
    ## -- KDTree for fast neighbor lookup
    tree = cKDTree(irh_coords)
    visited = set()  # Track visited pixels
    sequence = []  # Store ordered traversal

    current = leftmost
    while current is not None:
        sequence.append(current)
        visited.add(tuple(current))
        
        ## -- Query the nearest neighbors
        neighbors = tree.query_ball_point(current, r=1.5)  # Radius of 1.5 pixels for 8-connectivity
        neighbors = [irh_coords[i] for i in neighbors if tuple(irh_coords[i]) not in visited]
        
        ## -- Choose the next point that is to the right and not yet visited
        current = None
        for neighbor in sorted(neighbors, key=lambda x: x[1]):  # Sort by y-coordinate (left-to-right)
            if tuple(neighbor) not in visited:
                current = neighbor
                break
    
    return np.array(sequence)



def process_patch(filepath):
    """
    Process a mask file and return the traversal order for each IRH.
    """
    mask = load_mask(filepath)
    labeled_mask = separate_irhs(mask)

    irh_traversals = []
    for irh_label in range(1, labeled_mask.max() + 1):  # IRHs are labeled from 1 to max
        irh_coords = np.array(np.where(labeled_mask == irh_label)).T  # Extract IRH pixel coordinates
        leftmost = irh_coords[np.argmin(irh_coords[:, 1])]  # Find leftmost pixel
        traversal = traverse_irh(leftmost, irh_coords)
        irh_traversals.append(traversal)
    
    return irh_traversals





## -- Example usage
filepath = "./d_masks_64_64/20023150_patch_147_patch_2.csv"
irh_traversals = process_patch(filepath)

## -- Print traversal for each IRH
for i, traversal in enumerate(irh_traversals, start=1):
    print(f"IRH {i}: {traversal}")


# g = load_mask(filepath)
# plt.imshow(g)






#%%

## -- Generate Incremental Masks for an IRH

import numpy as np
from skimage.measure import label
from scipy.spatial import cKDTree


def generate_incremental_masks(mask, irh_label, increment_size=1):
    """
    Generate incremental masks for a single IRH.
    
    Parameters:
    - mask: the full binary mask
    - irh_label: the label of the IRH to process
    - increment_size: number of pixels to increment by in each step
                     (e.g., 10 means show 10 pixels, then 20, then 30, etc.)
    
    Returns:
    - List of binary masks, each showing progressively more pixels of the IRH
    """
    # Get the coordinates of the IRH
    labeled_mask = separate_irhs(mask)
    irh_coords = np.array(np.where(labeled_mask == irh_label)).T
    
    # Traverse the IRH to get an ordered sequence of pixels
    leftmost = irh_coords[np.argmin(irh_coords[:, 1])]  # Find leftmost pixel
    traversal = traverse_irh(leftmost, irh_coords)
    
    # Generate incremental masks
    incremental_masks = []
    total_pixels = len(traversal)
    
    # Generate masks showing progressively more pixels
    for pixel_count in range(increment_size, total_pixels + increment_size, increment_size):
        mask_copy = np.zeros_like(mask)
        # Limit pixel_count to not exceed total available pixels
        actual_pixels = min(pixel_count, total_pixels)
        
        for j in range(actual_pixels):
            x, y = traversal[j]
            mask_copy[x, y] = 1
        incremental_masks.append(mask_copy)
    
    # Always include a mask with all pixels if the last increment didn't include it
    if total_pixels % increment_size != 0:
        mask_copy = np.zeros_like(mask)
        for x, y in traversal:
            mask_copy[x, y] = 1
        incremental_masks.append(mask_copy)
    
    return incremental_masks



#%%

from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def unet_model_two_inputs(input_shape=(64, 64, 1)):
    """
    Defines a U-Net model that accepts two inputs: 
    1. The radargram patches.
    2. An additional input (e.g., dummy or meaningful data).

    Parameters:
    - input_shape: Shape of each input (default: (64, 64, 1)).

    Returns:
    - A Keras model accepting two inputs.
    """
    # Define the two inputs
    radargram_input = Input(shape=input_shape, name="radargram_input")
    additional_input = Input(shape=input_shape, name="additional_input")
    
    # Combine the inputs (e.g., via concatenation)
    combined_inputs = concatenate([radargram_input, additional_input], axis=-1)
    
    # Pass the combined inputs into the U-Net architecture
    x = unet_layers(combined_inputs)  # Modify unet_layers to take a tensor as input

    # Output layers (e.g., segmentation mask and termination map)
    mask_output = Conv2D(1, (1, 1), activation="sigmoid", name="mask_output")(x)
    termination_output = Conv2D(1, (1, 1), activation="sigmoid", name="termination_output")(x)

    # Create and return the model
    model = Model(inputs=[radargram_input, additional_input], outputs=[mask_output, termination_output])
    return model


def unet_layers(inputs):
    """
    Defines the U-Net architecture, starting from a given tensor.
    
    Parameters:
    - inputs: Input tensor.

    Returns:
    - Output tensor.
    """
    # Encoding path (downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    # Bottleneck layer
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)

    # Decoding path (upsampling)
    u3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c4)
    u3 = concatenate([u3, c3], axis=-1)

    u2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u3)
    u2 = concatenate([u2, c2], axis=-1)

    u1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u2)
    u1 = concatenate([u1, c1], axis=-1)

    # Output tensor
    outputs = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    return outputs






#%% ------ VISUALIZING


def visualize_samples(tf_dataset, num_samples=5):
    """
    Visualize radargrams and their corresponding masks side by side.

    Parameters:
    - tf_dataset: A TensorFlow dataset containing radargram and mask pairs.
    - num_samples: Number of samples to visualize.
    """
    samples_shown = 0  # Counter to track the number of samples shown

    # Iterate through batches in the dataset
    for radargrams, (masks, _) in tf_dataset:
        radargrams = radargrams.numpy()
        masks = masks.numpy()

        # Iterate over each sample in the batch
        for radargram, mask in zip(radargrams, masks):
            if samples_shown >= num_samples:
                return  # Stop after showing the required number of samples

            # Remove the channel dimension for visualization
            radargram = radargram.squeeze(axis=-1)
            mask = mask.squeeze(axis=-1)

            # Plot radargram and mask
            plt.figure(figsize=(10, 4))

            # Radargram
            plt.subplot(1, 2, 1)
            plt.imshow(radargram, cmap="gray")
            plt.title("Radargram")
            plt.axis("off")

            # Mask
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Mask")
            plt.axis("off")

            # Show the plot
            plt.tight_layout()
            plt.show()

            samples_shown += 1
            
            


#%%



## --- integrate incremental mask generation into the training data pipeline

import random
import tensorflow as tf

def load_radargram_and_mask(radargram_path, mask_path):
    """Load a radargram and the corresponding mask."""
    radargram = np.loadtxt(radargram_path, delimiter=',', dtype=float)  
    mask = load_mask(mask_path)
    return radargram, mask

def generate_training_data(
    radargram_path, mask_path, samples_per_irh, increment_size=1, enforce_zero_mask_ratio=40):
    """
    Generate training samples from a radargram and mask.

    Parameters:
    - radargram_path: path to the radargram file
    - mask_path: path to the mask file
    - increment_size: how many pixels to increment by, when creating partial masks
    - enforce_zero_mask_ratio: ensure 1 out of every `enforce_zero_mask_ratio` masks is all-zero
    - samples_per_irh: how many different training samples to generate per IRH (must be passed explicitly)
    """
    # if samples_per_irh is None:
    #     raise ValueError("samples_per_irh must be provided by the calling function.")

    radargram, mask = load_radargram_and_mask(radargram_path, mask_path)
    labeled_mask = separate_irhs(mask)
    irh_labels = np.unique(labeled_mask)[1:]  # Exclude background (0 label)
    
    training_samples = []
    counter = 0

    for irh_label in irh_labels:
        # Generate incremental masks with specified increment size
        incremental_masks = generate_incremental_masks(mask, irh_label, increment_size)
        
        # Ensure 1 out of every `enforce_zero_mask_ratio` samples is an all-zero mask
        zero_mask_interval = enforce_zero_mask_ratio
        for i in range(samples_per_irh):
            if (i + 1) % zero_mask_interval == 0:
                zero_mask = np.zeros_like(mask)
                training_samples.append((radargram, zero_mask))
                counter += 1
                print(f"Training sample {counter} generated (all-zero mask)")
            else:
                random_mask = random.choice(incremental_masks)
                training_samples.append((radargram, random_mask))
                counter += 1
                print(f"Training sample {counter} generated")
    
    return training_samples



#%%


import tensorflow as tf
import numpy as np
import os

def create_tf_data_pipeline(radargram_dir, mask_dir, samples_per_irh, batch_size):
    dataset = []
    
    for radargram_file in sorted(os.listdir(radargram_dir)):
        if radargram_file.endswith(".csv"):
            radargram_path = os.path.join(radargram_dir, radargram_file)
            mask_path = os.path.join(mask_dir, radargram_file)

            # Generate training samples
            training_samples = generate_training_data(
                radargram_path, mask_path, samples_per_irh=samples_per_irh
            )
            
            for radargram, mask in training_samples:
                # Add the channel dimension and ensure float32
                radargram = np.float32(radargram[..., np.newaxis])
                mask = np.float32(mask[..., np.newaxis])
                termination_mask = np.float32(np.zeros_like(mask))  # Dummy data for second output

                # Append in the required tuple structure
                dataset.append(
                    ((radargram, radargram), (mask, termination_mask))
                )
    
    def generator():
        for inputs, outputs in dataset:
            yield inputs, outputs

    # Convert to a TensorFlow dataset
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (  # Inputs
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
            ),
            (  # Outputs
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
            )
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)





#%%

import numpy as np



radargram_dir = "./d_grams_64_64/"
mask_dir = "./d_masks_64_64/"

# Define the number of samples per IRH and batch size
batch_size = 32
samples_per_irh = 10


# Create the dataset
tf_dataset = create_tf_data_pipeline(radargram_dir, mask_dir, samples_per_irh, batch_size)

"""
# Check the dataset structure
for data in tf_dataset.take(1):
    radargram_inputs, outputs = data
    radargram, dummy_additional = radargram_inputs
    mask, termination = outputs

    print("Radargram shape:", radargram.shape)
    print("Dummy additional input shape:", dummy_additional.shape)
    print("Mask shape:", mask.shape)
    print("Termination shape:", termination.shape)
"""

## -- shuffling the dataset globally
buffer_size = 20  # Adjust based on your dataset size for effective shuffling
tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)




# Count total samples
total_samples = sum(batch_size for batch in tf_dataset)
print(f"Total number of training samples: {total_samples}")


## -- Visualize samples
#visualize_samples(tf_dataset, num_samples=40)



##-- Define the model
model = unet_model_two_inputs(input_shape=(64, 64, 1))



# Compile the model
model.compile(
    optimizer="adam",
    loss={"mask_output": "binary_crossentropy", "termination_output": "binary_crossentropy"},
    metrics={"mask_output": "accuracy", "termination_output": "accuracy"}
)

# Print the model summary
model.summary()


# Add callbacks for better training control and monitoring
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Add the checkpoint callback
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='loss',  # Monitor training loss
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',  # Monitor training loss
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

####################################################################
## -- train the model
history = model.fit(
    tf_dataset,  # Your dataset should now return two inputs
    epochs=60,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the trained model
model.save('trained_model_more_data_emptymask_2input.keras')

####################################################################



# 5. Plot training history
plt.figure(figsize=(12, 4))
# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], "-*", label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['mask_output_accuracy'], "-*", label='Training Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()



#%% INFERENCE

"""

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

"""









































