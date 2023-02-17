import os
import cv2
import numpy as np


def readImages(images_path):
    """
    Reads the images in the folder specified in the given path.

    Parameters
    ----------
    images_path : string
        The path of the folder which contains the input images.
    
    Results
    -------
    image_array : array-like
        An array containing all the images inside the folder in array format.
    """

    print(f"Reading input images from '{images_path}'")
    image_list = []
    image_file_names = sorted(os.listdir(images_path))
    print(f"The files found: {image_file_names}")
    for file_name in image_file_names:
        image_path = images_path + "/" + file_name
        image = cv2.imread(image_path)
        if image is None:
            raise TypeError(f"Error loading {image_path}")
        else:
            image_list.append(image)
    
    image_array = np.array(image_list)
    return image_array

def getVMatrix(i, j, H):
    """
    Creates a V matrix which is to be used to find the B matrix in the initial parameter estimation.

    Parameters
    ----------
    i : int or float
        column index of homography matrix.
    j : int or float
        column index of homography matrix. 
    H : array-like
        The homography matrix.

    Results
    -------
    v : array-like
        The elements of V matrix.
    """
    v = np.array([H[0][i] * H[0][j], 
                    H[0][i] * H[1][j] + H[1][i] * H[0][j], 
                    H[1][i] * H[1][j], 
                    H[2][i] * H[0][j] + H[0][i] * H[2][j], 
                    H[2][i] * H[1][j] + H[1][i] * H[2][j], 
                    H[2][i] * H[2][j]])

    return v


def getBMatrix(H_matrices):
    """
    Estimates the B matrix from an array of homography matrices.

    Parameters
    ----------
    H_matrices : array-like
        An array containing all the homography matrices.
    
    Results
    -------
    B : array-like
        The B matrix.
    """
    print('\nEstimating the B matrix...')
    V_matrix = []

    for i, H in enumerate(H_matrices):
        V11 = getVMatrix(0, 0, H)
        V12 = getVMatrix(0, 1, H)
        V22 = getVMatrix(1, 1, H)
        V_matrix.append(V12)
        V_matrix.append(V11 - V22)
    V_matrix = np.array(V_matrix)
    
    # V obtained through last vector of Singular Value Decomposition (SVD) of V_matrix. SVD V_matrix = USV'
    U, S, V = np.linalg.svd(V_matrix, full_matrices=True)
    V = np.transpose(V)

    B = V[:, -1]

    print('\nEstimated B Matrix:\n', B)
    return B