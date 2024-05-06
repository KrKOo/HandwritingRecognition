import cv2
import random
import numpy as np
import imgaug as ia
from tqdm import tqdm
import multiprocessing
import imgaug.augmenters as iaa
from sklearn.utils import shuffle

def preprocess_images(images):
    # Scale images from [0, 1] to [0, 255]
    images_scaled_up = images * 255.0
    return images_scaled_up

def postprocess_images(images):
    # Scale images back from [0, 255] to [0, 1]
    images_scaled_down = np.clip(images, 0, 255) / 255.0
    return images_scaled_down
    
def create_speckle_noise(multiply_range, add_scale):
    return iaa.Sequential([
        iaa.MultiplyElementwise(multiply_range),
        iaa.AdditiveGaussianNoise(scale=add_scale)
    ])

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

    very_strong_gaussian = iaa.AdditiveGaussianNoise(scale=(0.2 * 255, 0.6 * 255))
    very_strong_poisson = iaa.AdditivePoissonNoise(lam=(30.0, 100.0), per_channel=True)
    very_strong_impulse = iaa.ImpulseNoise(p=(0.2, 0.5))
    very_strong_speckle = create_speckle_noise((0.75, 1.25), 0.2 * 255)

    return [
        slight_gaussian, slight_poisson, slight_impulse, slight_speckle,
        moderate_gaussian, moderate_poisson, moderate_impulse, moderate_speckle,
        strong_gaussian, strong_poisson, strong_impulse, strong_speckle,
        very_strong_gaussian, very_strong_poisson, very_strong_impulse, very_strong_speckle
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

class RandomComplexShapeErasing(iaa.Augmenter):
    def __init__(self, p=0.5, min_complexity=3, max_complexity=10, name=None, random_state=None, color=(0, 0, 0), transparency=0.5):
        super(RandomComplexShapeErasing, self).__init__(name=name, random_state=random_state)
        self.p = p
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.color = color
        self.transparency = transparency

    def _augment_images(self, images, random_state, parents, hooks):
        for i, image in enumerate(images):
            if random_state.rand() < self.p:
                num_vertices = random_state.randint(self.min_complexity, self.max_complexity)

                vertices = np.array([
                    [random_state.randint(0, image.shape[1]), random_state.randint(0, image.shape[0])]
                    for _ in range(num_vertices)
                ], dtype=np.int32)

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [vertices], 255)

                erasing_patch = np.full(image.shape, self.color, dtype=image.dtype)

                image = np.where(mask[:,:,None] == 255, np.round(erasing_patch * self.transparency + image * (1 - self.transparency)).astype(image.dtype), image)

                images[i] = image
        return images

    def __call__(self, image):
        return self._augment_images([image], self.random_state, None, None)[0]

    def get_parameters(self):
        return [self.p, self.min_complexity, self.max_complexity, self.color, self.transparency]

def process_batch(data_batch):
    images, labels, number = data_batch
    augmented_images = []
    augmented_labels = []

    blur_augmenters = create_blur_augmenters()
    noise_augmenters = create_noise_augmenters()
    random_erasing = RandomComplexShapeErasing(p=1.0,
                                               min_complexity=3,
                                               max_complexity=10,
                                               color=(0, 0, 0),
                                               transparency=0.5)


    for image, label in zip(images, labels):
        for i in range(number):
            augmented_images.append(image)
            augmented_labels.append(label)

            augmenter = random.choice(random.choice([noise_augmenters, blur_augmenters]))
            augmented = augmenter(image=image)
            augmented_images.append(augmented)
            augmented_labels.append(label)
        

    return np.array(augmented_images), np.array(augmented_labels)

def enrich_and_shuffle_dataset(images, labels, number=1, random_state=42, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    images = preprocess_images(images)
    num_images = len(images)
    chunk_size = num_images // num_processes
    data_batches = [(images[i:i + chunk_size], labels[i:i + chunk_size], number) for i in range(0, num_images, chunk_size)]

    results = []
    with multiprocessing.Pool(num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_batch, data_batches), total=len(data_batches), desc="Augmenting images: "):
            results.append(result)

    augmented_images, augmented_labels = zip(*results)
    augmented_images = np.concatenate(augmented_images)
    augmented_labels = np.concatenate(augmented_labels)
    
    augmented_images = postprocess_images(augmented_images)
    augmented_images, augmented_labels = shuffle(augmented_images, augmented_labels, random_state=random_state)

    return augmented_images, augmented_labels