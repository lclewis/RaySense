import numpy as np
import os
import os.path
from unittest import TestCase



#### ModelNet Data Processing Tool ####

modelnet_path = ' ' # path to the modelnet10 or 40

def read_off(filename: str):
    """
    function to read .off file as point clouds
    :param filename: file path
    :return:
    """
    f = open(filename)
    if 'OFF' != f.readline().strip():
        raise ValueError('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in f.readline().strip().split(' ')])
    verts = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.array(verts)


def read_ModelNet_file(folder: str = modelnet_path, index: int = 0):
    filenames = []
    with open(folder + 'test.txt', 'r') as f:
        for line in f:
            filenames.append(line.strip())

    cat = {}
    # with open(folder + 'modelnet10_id.txt', 'r') as f:
    with open(folder + 'modelnet40_id.txt', 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = int(ls[1])

    class_name = list(cat.keys())

    fn = filenames[index]
    cls = cat[fn.split('/')[0]]
    modelpath = os.path.join(modelnet_path, fn)
    point_set = read_off(modelpath)

    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # scale

    return point_set, cls, class_name


 

#### MNIST Data Processing Tool ####

def loadMNIST(prefix, folder):
    '''
    Function to load the MNIST Dataset
    :param prefix: (string) "train" or "t10k", corresponds to 60k training image/labels and 10k test images/labels
    :param folder: (string) path to the data folder
    :return: [data, label] data.shape = (N, 28, 28); N: number of data in the whole data set; label.shape = (N,)
    '''
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].copy().astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + "/" + prefix + '-labels-idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]
    return data, labels

def MNIST_dict(Images, Labels):
    '''
    Function to separate and store MNIST images in a dictionary
    :param Images: shape (N, 28, 28) data images
    :param Labels: shape (N, ) label of the corresponding image in Images
    :return Image_Dict: dictionary, Image_Dict[digit].shape = (n_digit, 28, 28) contains image collection for that digit
    '''

    ImageDict = {}

    for (i, digit) in enumerate(Labels):
        current_image = Images[i, :, :][np.newaxis, :]  # extract corresponding image with extra dimension added
        current_set = ImageDict.get(digit, current_image)  # get exsiting set
        current_set = np.concatenate((current_set, current_image))  # concatenate new member to existing set
        ImageDict[digit] = current_set  # update dictionary

    # Get rid of the repetition (the first and the second)
    for digit in range(10):
        ImageDict[digit] = ImageDict[digit][1:, :, :]

    return ImageDict

def vectorization(image_set):
    '''
    vectorize each image from an image set
    :param image_set: (N, d1, d2): numpy array consists of N images of shape d1xd2
    :return: image_set_vectorized: (d1xd2, N) each column stores the vectorized image
    '''

    # matrix to store vectorized image
    image_set_vectorized = np.zeros(
        [image_set.shape[1] * image_set.shape[2], image_set.shape[0]])

    for i in range(np.shape(image_set)[0]):
        image_set_vectorized[:, i] = np.reshape(image_set[i, :, :],
                                        [image_set.shape[1] * image_set.shape[2], ])
    return image_set_vectorized




class Test(TestCase):
    def test_readModelNet(self):
        point_set, cls = read_ModelNet_file(folder=modelnet_path, index=444)
        print(cls)
