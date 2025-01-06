

### ---------------------------------------------------------------------------
### author:             H.Moqadam
### date:               5.12.24
### desc:               This versionworks on the basis of IRHs seuqntial
###                     as masks.
### 
### ---------------------------------------------------------------------------

## -------- threading configuration 

import tensorflow as tf

# Set the number of threads TensorFlow can use
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


## ---- Traversal Logic

import os
import numpy as np
import numpy as np
from skimage.measure import label
from scipy.spatial import cKDTree

def load_mask(filepath):
    """Load a binary mask from a CSV file."""
    return np.loadtxt(filepath, delimiter=',', dtype=int)

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
    # KDTree for fast neighbor lookup
    tree = cKDTree(irh_coords)
    visited = set()  # Track visited pixels
    sequence = []  # Store ordered traversal

    current = leftmost
    while current is not None:
        sequence.append(current)
        visited.add(tuple(current))
        
        # Query the nearest neighbors
        neighbors = tree.query_ball_point(current, r=1.5)  # Radius of 1.5 pixels for 8-connectivity
        neighbors = [irh_coords[i] for i in neighbors if tuple(irh_coords[i]) not in visited]
        
        # Choose the next point that is to the right and not yet visited
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





# Example usage
filepath = "../DATA/masks/20023150_patch_16.csv"
irh_traversals = process_patch(filepath)

# Print traversal for each IRH
for i, traversal in enumerate(irh_traversals, start=1):
    print(f"IRH {i}: {traversal}")





#%%



## --- VISUALIZATION OF THE TRAVERSAL

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

# Load mask and process (reuse previous functions)
filepath = "../DATA/masks/20023150_patch_16.csv"
mask = np.loadtxt(filepath, delimiter=',', dtype=int)

def visualize_traversal(mask, traversal):
"""
    #Visualize the traversal order of an IRH step-by-step.
"""
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
import random
from skimage.measure import label
from scipy.spatial import cKDTree

def load_mask(filepath):
    """Load a binary mask from a CSV file."""
    return np.loadtxt(filepath, delimiter=',', dtype=int)

def separate_irhs(mask):
    """Separate individual IRHs using connected component labeling."""
    labeled_mask = label(mask, connectivity=2)  # 8-connectivity
    return labeled_mask

def traverse_irh(leftmost, irh_coords):
    """Traverse IRH pixels starting from the leftmost pixel."""
    tree = cKDTree(irh_coords)
    visited = set()  # Track visited pixels
    sequence = []  # Store ordered traversal

    current = leftmost
    while current is not None:
        sequence.append(current)
        visited.add(tuple(current))
        
        # Query the nearest neighbors
        neighbors = tree.query_ball_point(current, r=1.5)
        neighbors = [irh_coords[i] for i in neighbors if tuple(irh_coords[i]) not in visited]
        
        # Choose the next point that is to the right and not yet visited
        current = None
        for neighbor in sorted(neighbors, key=lambda x: x[1]):  # Sort by y-coordinate (left-to-right)
            if tuple(neighbor) not in visited:
                current = neighbor
                break
    
    return np.array(sequence)

def generate_incremental_masks(mask, irh_label, n=10):
    """Generate incremental masks for a single IRH."""
    # Get the coordinates of the IRH (1s in the mask)
    labeled_mask = separate_irhs(mask)
    irh_coords = np.array(np.where(labeled_mask == irh_label)).T
    
    # Traverse the IRH to get an ordered sequence of pixels
    leftmost = irh_coords[np.argmin(irh_coords[:, 1])]  # Find leftmost pixel
    traversal = traverse_irh(leftmost, irh_coords)
    
    # Generate incremental masks
    incremental_masks = []
    for i in range(1, len(traversal) + 1, n):  # Generate masks from 1 to n pixels
        mask_copy = np.zeros_like(mask)
        for j in range(i):
            x, y = traversal[j]
            mask_copy[x, y] = 1
        incremental_masks.append(mask_copy)
    
    return incremental_masks



#%%

## --- integrate incremental mask generation into the training data pipeline

import tensorflow as tf

def load_radargram_and_mask(radargram_path, mask_path):
    """Load a radargram and the corresponding mask."""
    radargram = np.loadtxt(radargram_path, delimiter=',', dtype=float)  # Assuming radargram is in .csv
    mask = load_mask(mask_path)
    return radargram, mask

def generate_training_data(radargram_path, mask_path, n=10):
    """Generate training data samples."""
    radargram, mask = load_radargram_and_mask(radargram_path, mask_path)
    
    # Separate IRHs and generate incremental masks for each IRH
    labeled_mask = separate_irhs(mask)
    irh_labels = np.unique(labeled_mask)[1:]  # Exclude background (0 label)
    
    training_samples = []
    
    # Counter to track how many samples are generated
    counter = 0
    
    # For each IRH in the mask, generate incremental masks
    for irh_label in irh_labels:
        incremental_masks = generate_incremental_masks(mask, irh_label, n)
        
        # Randomly select one of the incremental masks for training
        random_mask = random.choice(incremental_masks)
        
        # Create the training sample
        training_samples.append((radargram, random_mask))
        
        # Increment the counter and print the sample count
        counter += 1
        print(f"Training sample {counter} generated")
    
    return training_samples

def create_tf_data_pipeline(radargram_dir, mask_dir, batch_size=4):
    """Create TensorFlow data pipeline from the radargram and mask files."""
    dataset = []
    
    # Iterate over all radargram and mask files in the specified directories
    for radargram_file in sorted(os.listdir(radargram_dir)):
        if radargram_file.endswith(".csv"):
            radargram_path = os.path.join(radargram_dir, radargram_file)
            mask_path = os.path.join(mask_dir, radargram_file)  # Assuming masks have the same names as radargram files

            # Generate training samples
            training_samples = generate_training_data(radargram_path, mask_path, n=10)
            
            for radargram, mask in training_samples:
                dataset.append((radargram, mask))
    
    # Convert dataset to a TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset,
        (tf.float32, tf.float32)  # Radargram and mask are both float32
    )
    

    # Set parallelism options and batch the data
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1  # Use only one thread for the private thread pool
    options.experimental_threading.private_threadpool_size = 1  # Limit operations to 1 thread
    dataset = dataset.with_options(options)
    
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset



#%%

## --- the model

import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_shape=(512, 512, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder (downsampling)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x1_pool = layers.MaxPooling2D((2, 2))(x1)

    x2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1_pool)
    x2_pool = layers.MaxPooling2D((2, 2))(x2)

    # Bottleneck
    x3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x2_pool)

    # Decoder (upsampling)
    x4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x3)
    x4 = layers.Concatenate()([x4, x2])  # Concatenate with skip connection from encoder (x2)

    x5 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x4)
    x5 = layers.Concatenate()([x5, x1])  # Concatenate with skip connection from encoder (x1)

    # Output Layers
    irh_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='irh_output')(x5)  # IRH mask prediction
    terminate_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='terminate_output')(x5)  # Termination prediction

    # Define the model
    model = models.Model(inputs, [irh_output, terminate_output])

    return model




#%%


## -- compile the model

# Define the model
model = unet_model(input_shape=(512, 512, 1))

# Compile the model
model.compile(
    optimizer='adam',
    loss={
        'irh_output': 'binary_crossentropy',
        'terminate_output': 'binary_crossentropy',
    },
    metrics={
        'irh_output': ['accuracy'],
        'terminate_output': ['accuracy'],
    }
)

# Print model summary to verify architecture
model.summary()




#%%




## ---------------------



sample_count = sum(1 for _ in create_tf_data_pipeline(radargram_dir='../DATA/grams/', mask_dir='../DATA/masks/', batch_size=4))
print(f"Total training samples: {sample_count}")



## -- dynamically generate the data

train_data = create_tf_data_pipeline(radargram_dir='../DATA/grams/', 
                                     mask_dir='../DATA/masks/', 
                                     batch_size=4)
model.fit(train_data, epochs=20, verbose = 1)


print (" >>>>> THE TRAINING IS DONE <<<<<<")


















































