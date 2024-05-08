import cv2
import random
import numpy as np
import imgaug as ia
from tqdm import tqdm
import multiprocessing
from imgaug import augmenters as iaa
from sklearn.utils import shuffle

def preprocess_images(images):
    images_scaled_up = (images * 255).astype(np.uint8)
    return images_scaled_up

def postprocess_images(images):
    images_scaled_down = images.astype(np.float32) / 255.0
    return images_scaled_down
    
def create_speckle_noise(multiply_range, add_scale):
    return iaa.Sequential([
        iaa.MultiplyElementwise(multiply_range),
        iaa.AdditiveGaussianNoise(scale=add_scale)
    ])

def create_color_augmenter():
    return iaa.Sequential([
        iaa.Multiply((0.75, 1.25)),
        iaa.GammaContrast((0.5, 2.0)),
        iaa.AddToHueAndSaturation((-20, 20))
    ])

def create_elastic_deformation():
    return iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)

def apply_erode(image, iterations=1, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.float32)
    return cv2.erode(image, kernel, iterations=iterations)

def apply_dilate(image, iterations=1, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.float32)
    return cv2.dilate(image, kernel, iterations=iterations)

def create_flip_augmenters():
    horizontal_flip = iaa.Fliplr(1.0)
    vertical_flip = iaa.Flipud(0.5)
    return [horizontal_flip, vertical_flip]

def create_noise_augmenters():
    slight_gaussian = iaa.AdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255))
    slight_poisson = iaa.AdditivePoissonNoise(lam=(0, 16.0), per_channel=True)
    slight_impulse = iaa.ImpulseNoise(p=(0, 0.03))
    slight_speckle = create_speckle_noise((0.95, 1.05), 0.01 * 255)
    
    moderate_gaussian = iaa.AdditiveGaussianNoise(scale=(0.05 * 255, 0.1 * 255))
    moderate_poisson = iaa.AdditivePoissonNoise(lam=(10, 30.0), per_channel=True)
    moderate_impulse = iaa.ImpulseNoise(p=(0.03, 0.07))
    moderate_speckle = create_speckle_noise((0.9, 1.1), 0.05 * 255)
    
    strong_gaussian = iaa.AdditiveGaussianNoise(scale=(0.1 * 255, 0.2 * 255))
    strong_poisson = iaa.AdditivePoissonNoise(lam=(20, 60.0), per_channel=True)
    strong_impulse = iaa.ImpulseNoise(p=(0.07, 0.2))
    strong_speckle = create_speckle_noise((0.85, 1.15), 0.1 * 255)

    return [
        slight_gaussian, slight_poisson, slight_impulse, slight_speckle,
        moderate_gaussian, moderate_poisson, moderate_impulse, moderate_speckle,
        strong_gaussian, strong_poisson, strong_impulse, strong_speckle
    ]
    
def create_blur_augmenters():
    slight_gaussian_blur = iaa.GaussianBlur(sigma=(0.5, 1.0))
    moderate_gaussian_blur = iaa.GaussianBlur(sigma=(1.0, 2.0))

    slight_average_blur = iaa.AverageBlur(k=(2, 3))
    moderate_average_blur = iaa.AverageBlur(k=(3, 5))

    slight_median_blur = iaa.MedianBlur(k=3)
    moderate_median_blur = iaa.MedianBlur(k=5)

    return [
        slight_gaussian_blur, moderate_gaussian_blur, 
        slight_average_blur, moderate_average_blur, 
        slight_median_blur, moderate_median_blur
    ]

def create_all_augmenters():
    return (create_blur_augmenters(),
            create_noise_augmenters(),
            create_elastic_deformation(),
            create_flip_augmenters()
           )

def process_batch(data_batch):
    images, labels = data_batch
    augmented_images = []
    augmented_labels = []

    color_augmenter = create_color_augmenter()

    for image, label in zip(images, labels):
        eroded_image = apply_erode(image)
        dilated_image = apply_dilate(image)

        variants = np.array([image, eroded_image, dilated_image])
        variants = preprocess_images(variants)

        variants = np.concatenate((variants, [color_augmenter(image=variants[0])]), axis=0)
        variants = np.concatenate((variants, [color_augmenter(image=variants[1])]), axis=0)
        variants = np.concatenate((variants, [color_augmenter(image=variants[2])]), axis=0)

        blur_augmenters, noise_augmenters, elastic_def_aumgenter, flip_augmenter = create_all_augmenters()

        for variant in variants:
            augmented_images.append(variant)
            augmented_labels.append(label)
            
            current_augmenters = [elastic_def_aumgenter]
            current_augmenters.append(flip_augmenter[0])
            current_augmenters.append(flip_augmenter[1]) 
            current_augmenters.append(random.choice(blur_augmenters)) 
            current_augmenters.append(random.choice(noise_augmenters))  
            
            num_augmenters = random.randint(1, len(current_augmenters))
            selected_augmenters = random.sample(current_augmenters, num_augmenters)

            augmenter = iaa.Sequential(selected_augmenters)
            augmented_variant = augmenter(image=variant)
            augmented_images.append(augmented_variant)
            augmented_labels.append(label)
            
    augmented_images = postprocess_images(np.array(augmented_images))
    return augmented_images, np.array(augmented_labels)

def enrich_and_shuffle_dataset(images, labels, random_state=42, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    num_images = len(images)
    chunk_size = num_images // num_processes
    data_batches = [(images[i:i + chunk_size], labels[i:i + chunk_size]) for i in range(0, num_images, chunk_size)]

    results = []
    with multiprocessing.Pool(num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_batch, data_batches), total=len(data_batches), desc="Augmenting images: "):
            results.append(result)

    augmented_images, augmented_labels = zip(*results)
    augmented_images = np.concatenate(augmented_images)
    augmented_labels = np.concatenate(augmented_labels)
    
    augmented_images, augmented_labels = shuffle(augmented_images, augmented_labels, random_state=random_state)

    return augmented_images, augmented_labels