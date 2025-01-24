

### ---------------------------------------------------------------------------
### author:             H.Moqadam
### date:               21.1.25
### desc:               The double input model and dataset maker
### desc:               >> choosing small data and fixing the bugs. <<
### 
### ---------------------------------------------------------------------------






#%%

from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, models

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
    x = combined_inputs  # Start the U-Net from the combined inputs
    x = unet_layers(x)   # This function should define the U-Net layers (e.g., encoder-decoder structure)

    # Output layers (e.g., segmentation mask and termination map)
    mask_output = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="mask_output")(x)
    termination_output = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="termination_output")(x)

    # Create and return the model
    model = Model(inputs=[radargram_input, additional_input], outputs=[mask_output, termination_output])
    return model

from tensorflow.keras import layers, models

def unet_layers(input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoding path (downsampling)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Bottleneck layer
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # Decoding path (upsampling)
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)], axis=-1)
    
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)], axis=-1)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)], axis=-1)

    # Output layer with sigmoid activation to match target shape (64x64x1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



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
samples_per_irh = 20


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
buffer_size = 1000  # Adjust based on your dataset size for effective shuffling
tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)


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

################################## +++++++++++++++++++++++ ----------------....
## -- train the model
history = model.fit(
    tf_dataset,  # Your dataset should now return two inputs
    epochs=60,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the trained model
model.save('trained_model_more_data_emptymask_2input.keras')





#%% INFERENCE 


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random



# Load the trained model
saved_model = load_model('trained_model_more_data_emptymask_2input.keras')

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

# Predict using both inputs
predictions = saved_model.predict([radargram_input, additional_input])

# Process predictions
predicted_irh = predictions[0]
predicted_irh = normalize_matrix(predicted_irh, 0, 1)
prediction_stopping = predictions[1]
prediction_stopping = normalize_matrix(prediction_stopping, 0, 1)

# Display random numbers for the patch
print(f"The random number for the patch is {n} and second one {m}")


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
plt.imshow(predicted_irh[0,:,:,0], cmap="gray")
plt.subplot(224)
plt.title("Label")
plt.imshow(mask_piece_small, cmap="gray")
plt.tight_layout()
plt.show()



predicted_irh.shape
plt.figure("with matrix of all zeros")
b = predicted_irh[0, :, 0, 0]
c = b * (b<0.7)
plt.plot(c)
for_next_it = c * (c == c.max())
np.where(c == np.max(c))
plt.plot(for_next_it)



new_input = np.zeros_like(radargram_input)
new_input[0,:,0,0] = for_next_it



# Predict using both inputs
predictions = saved_model.predict([radargram_input, new_input])

# Process predictions
predicted_irh = predictions[0]
predicted_irh = normalize_matrix(predicted_irh, 0, 1)
prediction_stopping = predictions[1]
prediction_stopping = normalize_matrix(prediction_stopping, 0, 1)

# Display random numbers for the patch
print(f"The random number for the patch is {n} and second one {m}")


# Visualization
plt.figure("predictions2" , figsize=(10, 10))
plt.subplot(221)
plt.title("Radargram")
plt.imshow(gram_piece_small, cmap="gray")
plt.subplot(222)
plt.title("Mask")
plt.imshow(mask_piece_small, cmap="gray")
plt.subplot(223)
plt.title("Output - Predicted Horizon")
plt.imshow(predicted_irh[0, :, :, 0], cmap="gray")
plt.subplot(224)
plt.title("Output - Prediction Stopping")
plt.imshow(prediction_stopping[0, :, :, 0], cmap="gray")
plt.tight_layout()
plt.show()



b = predicted_irh[0, :, 0, 0]
c = b * (b<0.7)
plt.plot(c)
for_next_it = c * (c == c.max())
np.where(c == np.max(c))
plt.plot(for_next_it)

















