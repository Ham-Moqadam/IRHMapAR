

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


#%% loss fnctons

## -- custom softmax loss / Modified Softmax Function / smoothed softmax

from keras.saving import register_keras_serializable

@register_keras_serializable()
def custom_softmax_loss(y_true, y_pred):
    """
    Custom softmax loss function for IRH detection.
    """
    # Print shapes for debugging
    print("y_true shape:", tf.shape(y_true))
    print("y_pred shape:", tf.shape(y_pred))
    
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Apply softmax across spatial dimensions (height and width)
    # Reshape to combine batch and channel dimensions
    shape = tf.shape(y_pred)
    y_pred_reshaped = tf.reshape(y_pred, [-1, shape[1] * shape[2]])
    y_true_reshaped = tf.reshape(y_true, [-1, shape[1] * shape[2]])
    
    # Apply softmax
    spatial_softmax = tf.nn.softmax(y_pred_reshaped, axis=1)
    
    # Calculate cross entropy loss
    loss = tf.keras.losses.categorical_crossentropy(
        y_true_reshaped,
        spatial_softmax,
        from_logits=False
    )
    
    return tf.reduce_mean(loss)


def custom_softmax_loss(y_true, y_pred):
    logits = y_pred
    exp_logits = tf.exp(logits)
    probabilities = exp_logits / (tf.reduce_sum(exp_logits, axis=-1, keepdims=True) + 1.0)
    loss = -tf.reduce_sum(y_true * tf.math.log(probabilities + 1e-7), axis=-1)
    return tf.reduce_mean(loss)



import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def negative_likelihood_loss(y_true, y_pred, epsilon=1e-7):
    """
    Custom negative likelihood loss function for IRH prediction.
    
    Parameters:
    - y_true: Ground truth IRH mask
    - y_pred: Predicted IRH mask probabilities
    - epsilon: Small constant to prevent log(0)
    
    Returns:
    - Negative likelihood loss value
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip predictions to prevent numerical instability
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate negative log likelihood for positive pixels
    positive_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=[1, 2, 3])
    
    # Calculate negative log likelihood for negative pixels
    negative_loss = -tf.reduce_sum((1 - y_true) * tf.math.log(1 - y_pred), axis=[1, 2, 3])
    
    # Normalize by the number of pixels
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
    total_loss = (positive_loss + negative_loss) / num_pixels
    
    # Return mean loss across batch
    return tf.reduce_mean(total_loss)




# Optional: Weighted version of the loss function
@register_keras_serializable()
def weighted_negative_likelihood_loss(pos_weight=2.0):
    """
    Factory function to create a weighted negative likelihood loss.
    
    Parameters:
    - pos_weight: Weight for positive class (IRH pixels)
    
    Returns:
    - Weighted loss function
    """
    def loss(y_true, y_pred, epsilon=1e-7):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Weighted loss calculation
        positive_loss = -pos_weight * tf.reduce_sum(y_true * tf.math.log(y_pred), axis=[1, 2, 3])
        negative_loss = -tf.reduce_sum((1 - y_true) * tf.math.log(1 - y_pred), axis=[1, 2, 3])
        
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
        total_loss = (positive_loss + negative_loss) / num_pixels
        
        return tf.reduce_mean(total_loss)
    
    return loss




import tensorflow as tf

def dice_loss(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    y_true_f = tf.keras.backend.flatten(y_true)  # Flatten to a 1D array
    y_pred_f = tf.keras.backend.flatten(y_pred)  # Flatten to a 1D array
    intersection = tf.reduce_sum(y_true_f * y_pred_f)  # Intersection of prediction & truth
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


import keras as K
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0) errors
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy)
    return loss



import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


def combined_loss(y_true, y_pred):
    bce_loss = BinaryCrossentropy()(y_true, y_pred)
    dice_loss_value = dice_loss(y_true, y_pred)
    return 0.5 * bce_loss + 0.5 * dice_loss_value




#%%


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
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
                                     Conv2DTranspose, concatenate, 
                                     GlobalAveragePooling2D, Dense, Reshape, Lambda)




def unet_model_two_inputs(input_shape=(64, 64, 1)):
    """
    Modified U-Net model that accepts radargram and incremental label inputs.
    
    Parameters:
    - input_shape: Shape of each input (default: (64, 64, 1))
    
    Returns:
    - Keras model with two inputs and two outputs
    """
    # Define the two inputs
    radargram_input = Input(shape=input_shape, name="radargram_input")
    incremental_label_input = Input(shape=input_shape, name="incremental_label_input")
    
    # Combine the inputs
    combined_inputs = concatenate([radargram_input, incremental_label_input], axis=-1)
    
    # Encoding path (downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(combined_inputs)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)

    # Decoding path (upsampling)
    u3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c4)
    u3 = concatenate([u3, c3])

    u2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u3)
    u2 = concatenate([u2, c2])

    u1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u2)
    u1 = concatenate([u1, c1])

    # Output layers
    mask_output = Conv2D(1, (1, 1), activation="sigmoid", name="mask_output")(u1)
    termination_output = Conv2D(1, (1, 1), activation="sigmoid", name="termination_output")(u1)

    # Create and return the model
    model = Model(
        inputs=[radargram_input, incremental_label_input],
        outputs=[mask_output, termination_output]
    )
    return model



def unet_model_two_inputs_new_output(input_shape=(64, 64, 1)):
    """
    Modified U-Net model that accepts radargram and incremental label inputs.
    
    Parameters:
    - input_shape: Shape of each input (default: (64, 64, 1))
    
    Returns:
    - Keras model with two inputs and two outputs
    """
    # Define the two inputs
    radargram_input = Input(shape=input_shape, name="radargram_input")
    incremental_label_input = Input(shape=input_shape, name="incremental_label_input")
    
    # Combine the inputs
    combined_inputs = concatenate([radargram_input, incremental_label_input], axis=-1)
    
    # Encoding path (downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(combined_inputs)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)

    # Decoding path (upsampling)
    u3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c4)
    u3 = concatenate([u3, c3])

    u2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u3)
    u2 = concatenate([u2, c2])

    u1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u2)
    u1 = concatenate([u1, c1])

    # Mask output: (64, 1)
    mask_output = Conv2D(1, (1, 1), activation="sigmoid")(u1)  # (64, 64, 1)
    mask_output = tf.keras.layers.GlobalAveragePooling1D(name="mask_output")(mask_output)  # (64, 1)


    # Termination output: (1, 1)
    termination_output = GlobalAveragePooling2D()(u1)  # Convert (64,64,64) to (64,)
    termination_output = Dense(1, activation="sigmoid", name="termination_output")(termination_output)

    # Create and return the model
    model = Model(
        inputs=[radargram_input, incremental_label_input],
        outputs=[mask_output, termination_output]
    )
    return model







#%% ------ VISUALIZING


def visualize_samples(tf_dataset, num_samples=5):
    """
    Visualize samples from the dataset including radargrams, incremental labels, and masks.
    
    Parameters:
    - tf_dataset: TensorFlow dataset containing the samples
    - num_samples: Number of samples to visualize
    """
    samples_shown = 0

    for (radargrams, incremental_labels), (masks, _) in tf_dataset:
        radargrams = radargrams.numpy()
        incremental_labels = incremental_labels.numpy()
        masks = masks.numpy()

        for radargram, incremental_label, mask in zip(radargrams, incremental_labels, masks):
            if samples_shown >= num_samples:
                return

            # Remove channel dimension
            radargram = radargram.squeeze(axis=-1)
            incremental_label = incremental_label.squeeze(axis=-1)
            mask = mask.squeeze(axis=-1)

            plt.figure(figsize=(15, 4))

            # Radargram
            plt.subplot(1, 3, 1)
            plt.imshow(radargram, cmap="gray")
            plt.title("Radargram")
            plt.axis("off")

            # Incremental Label
            plt.subplot(1, 3, 2)
            plt.imshow(incremental_label, cmap="viridis")
            plt.title("Incremental Label")
            plt.axis("off")

            # Mask
            plt.subplot(1, 3, 3)
            plt.imshow(mask, cmap="gray")
            plt.title("Target Mask")
            plt.axis("off")

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
    radargram_path, mask_path, samples_per_irh, increment_size=1, enforce_zero_mask_ratio=15):
    """
    Generate training samples from a radargram and mask, including incremental labels.
    
    Parameters:
    - radargram_path: path to the radargram file
    - mask_path: path to the mask file
    - samples_per_irh: how many different training samples to generate per IRH
    - increment_size: how many pixels to increment by when creating partial masks
    - enforce_zero_mask_ratio: ensure 1 out of every n masks is all-zero
    
    Returns:
    - List of tuples (radargram, incremental_label, target_mask, termination_mask)
    """
    radargram, mask = load_radargram_and_mask(radargram_path, mask_path)
    labeled_mask = separate_irhs(mask)
    irh_labels = np.unique(labeled_mask)[1:]  # Exclude background (0 label)
    
    training_samples = []
    counter = 0

    for irh_label in irh_labels:
        # Generate incremental masks
        incremental_masks = generate_incremental_masks(mask, irh_label, increment_size)
        
        # Create corresponding incremental labels
        total_pixels = sum(np.sum(m) for m in incremental_masks)
        
        for i in range(samples_per_irh):
            if (i + 1) % enforce_zero_mask_ratio == 0:
                # Generate zero mask sample
                zero_mask = np.zeros_like(mask)
                incremental_label = np.zeros_like(mask)
                training_samples.append((radargram, incremental_label, zero_mask, zero_mask))
                counter += 1
                print(f"Training sample {counter} generated (all-zero mask)")
            else:
                # Select random incremental mask
                idx = random.randrange(len(incremental_masks))
                selected_mask = incremental_masks[idx]
                
                # Create corresponding incremental label
                incremental_label = np.zeros_like(mask)
                pixel_count = np.sum(selected_mask)
                incremental_label[selected_mask > 0] = pixel_count / total_pixels
                
                training_samples.append((radargram, incremental_label, selected_mask, np.zeros_like(mask)))
                counter += 1
                print(f"Training sample {counter} generated")
    
    return training_samples




#%%


import tensorflow as tf
import numpy as np
import os


def create_tf_data_pipeline(radargram_dir, mask_dir, samples_per_irh, batch_size):
    """
    Create TensorFlow dataset with radargram and incremental label inputs.
    
    Parameters:
    - radargram_dir: directory containing radargram files
    - mask_dir: directory containing mask files
    - samples_per_irh: number of samples to generate per IRH
    - batch_size: size of training batches
    
    Returns:
    - TensorFlow dataset
    """
    dataset = []
    
    for radargram_file in sorted(os.listdir(radargram_dir)):
        if radargram_file.endswith(".csv"):
            radargram_path = os.path.join(radargram_dir, radargram_file)
            mask_path = os.path.join(mask_dir, radargram_file)

            # Generate training samples
            training_samples = generate_training_data(
                radargram_path, mask_path, samples_per_irh=samples_per_irh
            )
            
            for radargram, incremental_label, mask, termination_mask in training_samples:
                # Add channel dimensions and ensure float32
                radargram = np.float32(radargram[..., np.newaxis])
                incremental_label = np.float32(incremental_label[..., np.newaxis])
                mask = np.float32(mask[..., np.newaxis])
                termination_mask = np.float32(termination_mask[..., np.newaxis])
                
                dataset.append(
                    ((radargram, incremental_label), (mask, termination_mask))
                )
    
    def generator():
        for inputs, outputs in dataset:
            yield inputs, outputs

    # Convert to TensorFlow dataset
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (  # Inputs
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32)
            ),
            (  # Outputs
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32)
            )
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)






#%%

import numpy as np



radargram_dir = "./d_grams_64_64/"
mask_dir = "./d_masks_64_64/"

# Define the number of samples per IRH and batch size
batch_size = 32
samples_per_irh = 50


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
buffer_size = 50  # Adjust based on your dataset size for effective shuffling
tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)




# Count total samples
total_samples = sum(batch_size for batch in tf_dataset)
print(f"Total number of training samples: {total_samples}")


## -- Visualize samples
#visualize_samples(tf_dataset, num_samples=40)



##-- Define the model
model = unet_model_two_inputs(input_shape=(64, 64, 1))



# Compile the model
# Using the basic negative likelihood loss

model.compile(
    optimizer="adam",
    loss={
        "mask_output": "binary_crossentropy",
        "termination_output": "binary_crossentropy"
    },
    metrics={
        "mask_output": "accuracy",
        "termination_output": "accuracy"
    }
)

"""

# Or using the weighted version (if you want to give more importance to IRH pixels)
weighted_loss = weighted_negative_likelihood_loss(pos_weight=2.0)
model.compile(
    optimizer="adam",
    loss={
        "mask_output": weighted_loss,
        "termination_output": "binary_crossentropy"
    },
    metrics={
        "mask_output": "accuracy",
        "termination_output": "accuracy"
    }
)
"""

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

"""
## -- Optional: Add learning rate scheduler
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate
)
"""



####################################################################
## -- train the model
history = model.fit(
    tf_dataset,  # Your dataset should now return two inputs
    epochs=90,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the trained model
model.save('2input_BCE_large_ds.keras')

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









































