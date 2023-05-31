from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from scipy import linalg

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


def load_Real_data(img_height, img_width, batch_size):
    data_folder = 'C:/~data path/'
    print(data_folder)

    # 하지만, 사용하는 데이터가 이미 [0, 1] 값만 가짐, [-1,1]로 정규화! ( 256 곱 후에, -127.5 후에, / 127.5 )
    data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5)
    x_train = data_gen.flow_from_directory(directory=data_folder
                                           , target_size=(img_height, img_width)
                                           , batch_size=batch_size
                                           , shuffle=True
                                           , class_mode='input'
                                           , color_mode = 'rgb'
                                           , subset="training"
                                           )
    return x_train

def load_Fake_data(img_height, img_width, batch_size):
    data_folder = 'C:/~data path/'
    print(data_folder)

    # 하지만, 사용하는 데이터가 이미 [0, 1] 값만 가짐, [-1,1]로 정규화! ( 256 곱 후에, -127.5 후에, / 127.5 )
    data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5)
    x_train = data_gen.flow_from_directory(directory=data_folder
                                           , target_size=(img_height, img_width)
                                           , batch_size=batch_size
                                           , shuffle=True
                                           , class_mode='input'
                                           , color_mode = 'rgb'
                                           , subset="training")
    return x_train

def compute_embeddings(dataloader, count): # Calc Embeddings
    image_embeddings = []

    for _ in tqdm(range(count)):
        images = dataloader
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)


def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    # calculate sum squared difference between means


    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean) #trace: 대각 합
    return fid


BATCH_SIZE = 256
count = math.ceil(10000/BATCH_SIZE)
inception_model = tf.keras.applications.InceptionV3(include_top=False,
                              weights="imagenet",
                              pooling='avg')


# compute embeddings for real images
real_image_embeddings = compute_embeddings(load_Real_data(100,80,BATCH_SIZE), count)

# compute embeddings for generated images
generated_image_embeddings = compute_embeddings(load_Fake_data(100,80,BATCH_SIZE), count)
print(real_image_embeddings.shape  , generated_image_embeddings.shape)
fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
print(fid)

