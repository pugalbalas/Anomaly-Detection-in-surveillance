import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Set parameters
IMG_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 30
SAVE_INTERVAL = 500
SAVE_DIR = "processed_"
MODEL_SAVE_PATH = "gan_model.h5"

os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset
def load_images(dataset_path):
    image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))  # Load all .jpg files
    if len(image_files) == 0:
        raise ValueError(f"No images found in the dataset path: {dataset_path}")
    
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    
    def load_and_preprocess(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        return image
    
    dataset = dataset.map(load_and_preprocess).batch(BATCH_SIZE)
    return dataset

# Build Generator
def build_generator():
    model = Sequential([
        Dense(8 * 8 * 256, input_dim=LATENT_DIM),
        Reshape((8, 8, 256)),
        Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu'),
        Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu'),
        Conv2DTranspose(CHANNELS, (4, 4), strides=2, padding='same', activation='tanh')
    ])
    return model

# Build Discriminator
def build_discriminator():
    model = Sequential([
        Conv2D(64, (4, 4), strides=2, padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Conv2D(128, (4, 4), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Compile GAN
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(LATENT_DIM,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

# Train GAN
def train(dataset_path):
    dataset = load_images(dataset_path)
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))

    for epoch in range(EPOCHS):
        for real_images in dataset:
            batch_size = real_images.shape[0]
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            fake_images = generator.predict(noise)
            
            d_loss_real = discriminator.train_on_batch(real_images, real_labels[:batch_size])
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels[:batch_size])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            g_loss = gan.train_on_batch(noise, real_labels[:batch_size])

        if epoch % SAVE_INTERVAL == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss[0]}, G Loss: {g_loss}")
            save_generated_images(generator, epoch)
    
    generator.save(MODEL_SAVE_PATH)

# Save generated images
def save_generated_images(generator, epoch):
    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    gen_image = generator.predict(noise)[0]
    gen_image = (gen_image * 127.5 + 127.5).astype(np.uint8)
    img_path = os.path.join(SAVE_DIR, f"generated_{epoch}.png")
    tf.keras.utils.save_img(img_path, gen_image)

if __name__ == "_main_":
    dataset_path = "./processed_test_img"  # Change this to your dataset folder
    train(dataset_path)