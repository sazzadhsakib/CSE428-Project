from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot




def load_images(path, size=(256, 512)):
    src_list, tar_list = list(), list()

    for filename in listdir(path):

        pixels = load_img(path + filename, target_size=size)

        pixels = img_to_array(pixels)

        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)
    return [asarray(src_list), asarray(tar_list)]



path = 'data/train/'

[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)

n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_images[i].astype('uint8'))

for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()



from pix2pix_model import define_discriminator, define_generator, define_gan, train


image_shape = src_images.shape[1:]

d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

gan_model = define_gan(g_model, d_model, image_shape)

data = [src_images, tar_images]


def preprocess_data(data):

    X1, X2 = data[0], data[1]

    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


dataset = preprocess_data(data)

from datetime import datetime

start1 = datetime.now()

train(d_model, g_model, gan_model, dataset, n_epochs=10, n_batch=1)

stop1 = datetime.now()

execution_time = stop1 - start1
print("Execution time is: ", execution_time)


from keras.models import load_model
from numpy.random import randint

model = load_model('saved_model_10epochs.h5')



def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']

    for i in range(len(images)):

        pyplot.subplot(1, 3, 1 + i)

        pyplot.axis('off')

        pyplot.imshow(images[i])

        pyplot.title(titles[i])
    pyplot.show()


[X1, X2] = dataset

ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

gen_image = model.predict(src_image)

plot_images(src_image, gen_image, tar_image)

