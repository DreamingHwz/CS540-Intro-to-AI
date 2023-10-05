from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def multi_return ():
    return "a string ", 5
my_string , my_int = multi_return ()

def load_and_center_dataset(filename):
    x = np.load(filename)
    center = x - np.mean(x, axis = 0)
    return center

def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset)/(len(dataset)-1)

def get_eig(S, m):
    evalue, evector = eigh(S)
    return np.diag(evalue[-1:-1-m:-1]), evector[:, -1:-1-m:-1]

def get_eig_prop(S, prop):
    sum = np.trace(np.diag(eigh(S, eigvals_only=True)))
    evalue, evector = eigh(S, subset_by_value=[prop*sum, sum])
    return np.diag(evalue[::-1]), evector[:, ::-1]

def project_image(image, U):
    d = len(image)
    projection = np.array([0.0]*d)
    for m in range(len(U[0])):
        a = np.dot(U[:, m], image)
        projection += a * U[:, m]
    return projection

def display_image(orig, proj):
    orig = orig.reshape(32, 32).transpose()
    proj = proj.reshape(32, 32).transpose()
    
    plt.figure()
    plt.subplot(1, 2, 1); plt.title("Original")
    im = plt.imshow(orig, aspect = 'equal')
    plt.colorbar(im)

    plt.subplot(1, 2, 2); plt.title("Projection")
    im = plt.imshow(proj, aspect = 'equal')
    plt.colorbar(im)
    plt.show()
    return

