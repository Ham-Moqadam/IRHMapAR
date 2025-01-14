

### ---------------------------------------------------------------------------
### author:             H.Moqadam
### date:               5.12.24
### desc:               This versionworks on the basis of IRHs seuqntial
###                     as masks.
### new date:           6.1.2025
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

"""

## --- VISUALIZATION OF THE TRAVERSAL



import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

# Load mask and process (reuse previous functions)
filepath = "../DATA/masks/20023150_patch_25.csv"
mask = np.loadtxt(filepath, delimiter=',', dtype=int)

def visualize_traversal(mask, traversal):
    
    #Visualize the traversal order of an IRH step-by-step.
    
    plt.figure(figsize=(12,12))
    plt.imshow(mask, cmap='gray')  # Display the mask
    for i, point in enumerate(traversal):
        plt.scatter(point[1], point[0], color='red', s=10)  # Plot traversal point
        plt.title(f"Step {i + 1}: Point {point}")
        plt.pause(0.01)  # Pause to create animation effect
    plt.show()



# Process the mask and extract traversal for IRH 21 (example)
labeled_mask = label(mask, connectivity=2)
irh_coords = np.array(np.where(labeled_mask == 6)).T  # Replace 21 with desired IRH
leftmost = irh_coords[np.argmin(irh_coords[:, 1])]  # Find leftmost pixel
tree = cKDTree(irh_coords)

# Traverse IRH 21
visited = set()
sequence = []
current = leftmost
while current is not None:
    sequence.append(current)
    visited.add(tuple(current))
    neighbors = tree.query_ball_point(current, r=1.5)
    neighbors = [irh_coords[i] for i in neighbors if tuple(irh_coords[i]) not in visited]
    current = None
    for neighbor in sorted(neighbors, key=lambda x: x[1]):
        if tuple(neighbor) not in visited:
            current = neighbor
            break

sequence = np.array(sequence)

# Visualize the traversal
visualize_traversal(mask, sequence)

"""



#%%


## --- DYNAMIC MASK GENERATON

        ## -- generate incremental msk (n = 10)
        ## -- dynamic mask generation 


#%%

## -- Generate Incremental Masks for an IRH

import numpy as np
from skimage.measure import label
from scipy.spatial import cKDTree


def generate_incremental_masks(mask, irh_label, increment_size=5):
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

## --- integrate incremental mask generation into the training data pipeline

import random
import tensorflow as tf

def load_radargram_and_mask(radargram_path, mask_path):
    """Load a radargram and the corresponding mask."""
    radargram = np.loadtxt(radargram_path, delimiter=',', dtype=float)  
    mask = load_mask(mask_path)
    return radargram, mask

def generate_training_data(radargram_path, mask_path, increment_size=5, samples_per_irh=20):
    """
    Generate training samples from a radargram and mask.
    
    Parameters:
    - radargram_path: path to the radargram file
    - mask_path: path to the mask file
    - increment_size: how many pixels to increment by, when creating partial masks
    - samples_per_irh: how many different training samples to generate per IRH
    """
    radargram, mask = load_radargram_and_mask(radargram_path, mask_path)
    
    labeled_mask = separate_irhs(mask)
    irh_labels = np.unique(labeled_mask)[1:]  # Exclude background (0 label)
    
    training_samples = []
    counter = 0
    
    for irh_label in irh_labels:
        # Generate incremental masks with specified increment size
        incremental_masks = generate_incremental_masks(mask, irh_label, increment_size)
        
        # Take the specified number of samples from these masks
        for _ in range(samples_per_irh):
            random_mask = random.choice(incremental_masks)
            training_samples.append((radargram, random_mask))
            counter += 1
            print(f"Training sample {counter} generated")
    
    return training_samples




def create_tf_data_pipeline(radargram_dir, mask_dir, batch_size=4):
    """Create TensorFlow data pipeline from the radargram and mask files."""

    dataset = []
    
    for radargram_file in sorted(os.listdir(radargram_dir)):
        if radargram_file.endswith(".csv"):
            radargram_path = os.path.join(radargram_dir, radargram_file)
            mask_path = os.path.join(mask_dir, radargram_file)

            training_samples = generate_training_data(radargram_path, mask_path, samples_per_irh=20)
            
            for radargram, mask in training_samples:
                # Convert to float32 and add channel dimension
                radargram = np.float32(radargram[..., np.newaxis])
                mask = np.float32(mask[..., np.newaxis])
                termination_mask = np.float32(np.zeros_like(mask))
                
                # Create the nested tuple structure explicitly
                output_tuple = (mask, termination_mask)
                dataset.append((radargram, output_tuple))
    
    
    
    def generator():
        for radargram, (mask, term) in dataset:
            # Ensure we yield with the correct nested structure
            yield (radargram, (mask, term))

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32)
            )
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    

    ## -- Set parallelism options and batch the data
    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = 1  # Use only one thread for intra-op parallelism
    options.threading.private_threadpool_size = 1  # Limit operations to 1 thread
    dataset = dataset.with_options(options)

    
    
    ## -- Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset



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

# Assuming the TensorFlow dataset is created with your function:
radargram_dir = "./d_grams_64_64/"
mask_dir = "./d_masks_64_64/"
batch_size = 4


# Create the dataset
tf_dataset = create_tf_data_pipeline(radargram_dir, mask_dir, batch_size=batch_size)

# Visualize samples
visualize_samples(tf_dataset, num_samples=40)




def count_total_samples(tf_dataset):
    """
    Count the total number of samples in a TensorFlow dataset.
    
    Parameters:
    - tf_dataset: A TensorFlow dataset.
    
    Returns:
    - Total number of samples.
    """
    total_samples = 0
    for batch in tf_dataset:
        batch_size = batch[0].shape[0]  # Get the batch size
        total_samples += batch_size
    return total_samples





#%%

## --- the model

import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_shape=(64, 64, 1)):
    """Defines a U-Net for 64x64 input patches."""
    inputs = layers.Input(shape=input_shape)

    # Encoder (downsampling)
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1_pool = layers.MaxPooling2D((2, 2))(x1)

    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1_pool)
    x2_pool = layers.MaxPooling2D((2, 2))(x2)

    x3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x2_pool)
    x3_pool = layers.MaxPooling2D((2, 2))(x3)

    # Bottleneck
    x4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x3_pool)

    # Decoder (upsampling)
    x5 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x4)
    x5 = layers.Concatenate()([x5, x3])  # Skip connection

    x6 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x5)
    x6 = layers.Concatenate()([x6, x2])  # Skip connection

    x7 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x6)
    x7 = layers.Concatenate()([x7, x1])  # Skip connection

    # Output layers
    irh_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='irh_output')(x7)  # IRH mask prediction
    terminate_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='terminate_output')(x7)  # Termination prediction

    # Define the model
    model = models.Model(inputs, [irh_output, terminate_output])
    return model


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



#%%


## -- total number of traing samples
sample_count = sum(1 for _ in create_tf_data_pipeline(radargram_dir='./d_grams_64_64/', mask_dir='./d_masks_64_64/', batch_size=4))
print(f"Total training samples: {sample_count}")
# print(f"Training sample {counter} generated from file {radargram_path}, IRH {irh_label}")




# 5. Optional: Add learning rate scheduler
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate
)


##-- Define the model
model = unet_model(input_shape=(64, 64, 1))




# Debug: Count total samples
total_samples = 0
batch_count = 0

## -- dynamically generate the data (Create dataset)
train_data = create_tf_data_pipeline(radargram_dir='./d_grams_64_64/', 
                                   mask_dir='./d_masks_64_64/', 
                                   batch_size=4)


## -- to see the number of training data (Count the total number of samples in your training dataset)
total_samples = count_total_samples(train_data)
print(f"Total number of training samples: {total_samples}")


# Debug: Test a single batch
for inputs, (mask, term) in train_data.take(1):
    print("Input shape:", inputs.shape)
    print("Mask shape:", mask.shape)
    print("Termination shape:", term.shape)
    break

# Count samples in each batch
for batch_x, (batch_mask, batch_term) in train_data:
    batch_count += 1
    samples_in_batch = batch_x.shape[0]  # Should be 4 (or less for last batch)
    total_samples += samples_in_batch
    print(f"Batch {batch_count}: {samples_in_batch} samples")

print(f"\nTotal batches: {batch_count}")
print(f"Total samples: {total_samples}")




#%%

## -- training block

# 1. Add callbacks for better training control and monitoring
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss instead of training loss
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',  # Monitor validation loss instead of training loss
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss instead of training loss
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 2. Split data into training and validation sets
train_size = int(0.8 * total_samples)  # 80% for training
train_dataset = train_data.take(train_size)
val_dataset = train_data.skip(train_size)

# 3. Compile the model (ensure loss and metrics are defined)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss={
        'irh_output': 'binary_crossentropy',
        'terminate_output': 'binary_crossentropy',
    },
    metrics={
        'irh_output': ['accuracy'],
        'terminate_output': ['accuracy'],
    }
)

# 4. Enhanced training call
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)




###############################################################################

## -- old train data
#

# Update model compilation to use the custom loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss={
        'irh_output': 'binary_crossentropy',
        'terminate_output': 'binary_crossentropy',
    },
    metrics={
        'irh_output': ['accuracy'],
        'terminate_output': ['accuracy'],
    }
)

 
# Add the checkpoint callback
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='loss',
    save_best_only=True,
    verbose=1
)


# Train the model
history = model.fit(
    train_data,
    epochs=80,
    verbose=1,
    callbacks=[model_checkpoint]
)


# Save the trained model
model.save('trained_model.keras')




# 5. Plot training history
plt.figure(figsize=(12, 4))
# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()







#%%


## -- OLD PRWEDICITON WITH THEM 

from tensorflow.keras.models import load_model

# Load the saved model
saved_model = load_model('trained_model.keras')


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


# plt.savefig("./res_3_BC.png")


## -----------------------------------------------------------------------
## -- NEW PREDICTION (ITERATIVELY)


import numpy as np
import tensorflow as tf



def trace_horizon(model, radargram, start_pixel, max_steps=100, terminate_threshold=0.9):
    """
    Traces a single horizon in a radargram using an iterative process.
    Args:
    - model: Trained model with irh prediction and termination outputs
    - radargram: Input radargram as a numpy array of shape (H, W)
    - start_pixel: Tuple (x, y) indicating the starting pixel
    - max_steps: Maximum number of iterations for tracing
    - terminate_threshold: Termination signal threshold
    Returns:
    - horizon: List of traced pixels [(x1, y1), (x2, y2), ...]
    """
    current_pixel = start_pixel
    horizon = [current_pixel]  # Initialize the horizon with the start pixel
    H, W = radargram.shape
    
    for step in range(max_steps):
        # Prepare the input for the model
        input_data = prepare_input(radargram, horizon, H, W)
        
        # Make predictions - model returns a list of outputs
        predictions = model.predict(input_data)
        
        # First output is irh_output, second is terminate_output
        irh_output = predictions[0].squeeze()  # Remove batch dimension
        terminate_output = predictions[1].squeeze()  # Remove batch dimension
        
        # Handle termination output
        if np.ndim(terminate_output) > 0:  # If terminate_output is a map
            terminate_prob = terminate_output.max()  # Use the maximum value
        else:
            terminate_prob = float(terminate_output)  # Convert to float if scalar
            
        # Check for termination
        if terminate_prob >= terminate_threshold:
            print(f"Terminated after {step + 1} steps at pixel {current_pixel}")
            break
            
        # Find the next pixel (highest probability in irh_output)
        next_pixel = np.unravel_index(np.argmax(irh_output), (H, W))
        
        # Validate the next pixel
        if (
            next_pixel in horizon or  # Already visited
            not (0 <= next_pixel[0] < H and 0 <= next_pixel[1] < W)  # Out of bounds
        ):
            print(f"Stopped due to invalid next pixel: {next_pixel}")
            break
            
        # Add the predicted pixel to the list
        horizon.append(next_pixel)
        current_pixel = next_pixel
        
    return horizon



def prepare_input(radargram, horizon, H, W):
    """
    Prepares input data for the model with encoded horizon pixels.
    Args:
    - radargram: Input radargram of shape (H, W)
    - horizon: List of traced pixels [(x1, y1), (x2, y2), ...]
    - H, W: Dimensions of the radargram
    Returns:
    - input_data: Model-ready input of shape (1, H, W, 1)
    """
    # Create a copy of the radargram to avoid modifying the original
    input_data = radargram.copy()
    
    # Create a mask for the horizon
    horizon_mask = np.zeros_like(input_data)
    for x, y in horizon:
        horizon_mask[x, y] = 1.0  # Highlight horizon pixels
    
    # Stack the original radargram and horizon mask
    input_data = np.stack([input_data, horizon_mask], axis=-1)
    
    return np.expand_dims(input_data, axis=0)  # Add batch dimension



import numpy as np
import tensorflow as tf

# First define the two functions (trace_horizon and prepare_input) from earlier

# Now let's trace horizons from your chosen starting points
model = tf.keras.models.load_model('trained_model.keras')
radargram = gram_piece_small

# Define your starting points as (row, column) coordinates
starting_points = [
    (12, 0),  # first starting point
    (25, 0),  # second starting point
    (40, 0)   # third starting point
]

# Trace horizons from each starting point
traced_horizons = []
for start_point in starting_points:
    horizon = trace_horizon(model, radargram, start_point, 
                          max_steps=100,    # adjust based on your image width
                          terminate_threshold=0.9)  # adjust if needed
    traced_horizons.append(horizon)
    print(f"Completed tracing horizon from starting point {start_point}")

# Now traced_horizons is a list where each element is a list of points for one horizon












"""


# Example usage:
def trace_full_image(model, radargram, start_row=0, step_size=10):
    #   
    # Traces horizons across the full image starting from multiple points.
    # Args:
    # - model: Trained model
    # - radargram: Input radargram
    # - start_row: Row to start tracing from
    # - step_size: Horizontal spacing between start points
    # Returns:
    # - List of traced horizons
    # 
    horizons = []
    H, W = radargram.shape
    
    # Generate starting points across the image
    for col in range(0, W, step_size):
        start_pixel = (start_row, col)
        horizon = trace_horizon(model, radargram, start_pixel)
        horizons.append(horizon)
    
    return horizons

"""


































































