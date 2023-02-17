import os
import cv2
import math
import numpy as np
import argparse, textwrap
from scipy import optimize
from Code.Utils.autocalib_utils import readImages, getBMatrix

debug = False

class AutoCalibrator():

    def __init__(self, images, pattern_size, square_size):
        """
        Performs auto-calibration on a monocular camera based on a number of images taken from it.

        Parameters
        ----------
        images_path : string
            The path of folder containing the images.
        pattern_size : tuple of ints
            Tuple containing number of rows and columsn of the chessboard pattern.
        square_size : float
            The length of the side of a square in the chessboard.

        """
        self.images = images
        self.nrows = int(pattern_size[0])
        self.ncols = int(pattern_size[1])
        self.square_size = square_size
    


    def getChessBoardCorners(self):
        """
        Gives the chessboard correspondences between world points and image points.
        
        Results
        -------
        world_pts_array : array-like
            An array containing all the world points in the images.
        image_pts_array : array-like
            An array containing all the image points in the images.
        """
        
        world_points = np.zeros((self.nrows * self.ncols, 3), np.float32)
        world_points[:, : 2] = np.mgrid[0: self.nrows, 0: self.ncols].T.reshape(-1, 2)
        world_points *= self.square_size

        world_pts_list = []
        image_pts_list = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
        count = 0
        for image in self.images:
            
            print(f'\nFinding Chessboard corners of image {count+1}')
            img = image.copy()
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_img, patternSize=(self.nrows, self.ncols))

            if ret:
                print(f"\nChessboard Detected in image {count+1}")
                count += 1
                corners_dash = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
                corners = corners.reshape(-1, 2)
                world_pts_list.append(world_points)
                image_pts_list.append(corners_dash)

            if debug:
                if not os.path.exists('./Code/Debug'):
                    os.mkdir('./Code/Debug')
                cv2.drawChessboardCorners(img, (self.nrows, self.ncols), corners, ret)
                cv2.imwrite('./Code/Debug/' + str(count) + ".png", img)

        world_pts_array = np.array(world_pts_list)
        image_pts_array = np.array(image_pts_list)
        world_pts_array = np.reshape(world_pts_array, (count, self.ncols*self.nrows, 3))
        image_pts_array = np.reshape(image_pts_array, (count, self.ncols*self.nrows, 2))

        return image_pts_array, world_pts_array



    def getInitialParameterEstimates(self, image_points, world_points):
        """
        Estimates all the initial parameters which can be fed into non-linear optimizer.

        Parameters
        ----------
        world_pts_array : array-like
            An array containing all the world points in the images.
        image_pts_array : array-like
            An array containing all the image points in the images.
        
        Results
        -------
        initial_parameters : dict
            Dictionary containing initial estimates of all parameters. 
        """

        NImages = len(world_points)

        print("\nEstimating the homography Matrices...")
        H_matrices = []

        # get the homography between world points and image points in all the images
        for i in range(NImages):
            
            H = self.getHomography(image_points[i], world_points[i])
            # print(H)
            H_matrices.append(H)
        H_matrices = np.array(H_matrices)

        # get the initial camera intrinsic matrix
        K_matrix_initial = self.getInitKMatrix(H_matrices)
        print("\nInitial Camera Intrinsic Matrix: \n", K_matrix_initial)

        # get the approximate distortion
        k_initial = self.getInit_k()
        print("\nInitial Approximate Radial Distortion: \n", k_initial)

        # get the camera extrinsics
        RT_array = self.getExtrinsicParameters(K_matrix_initial, H_matrices)

        return K_matrix_initial, k_initial, RT_array


    def getHomography(self, image_points, world_points):
        """
        Estimates homography between world points and image points in an image.

        Parameters
        ----------
        world_points : array-like
            An array containing all the world points in an image.
        image_points : array-like
            An array containing all the image points in an image.
        
        Results
        -------
        H : array-like
            A 3 x 3 homography matrix which relates the world points and image points.
        """

        Npoints = len(image_points)
        A = np.zeros((2 * Npoints, 9), np.float32)

        for i in range(Npoints):

            # get world co-ordinates
            x = world_points[i][0]
            y = world_points[i][1]
            # get image co-ordinates
            x_dash = image_points[i][0]
            y_dash = image_points[i][1]

            row1 = np.array([-x, -y, -1, 0, 0, 0, x * x_dash, y * x_dash, x_dash])
            row2 = np.array([0, 0, 0, -x, -y, -1, x * y_dash, y * y_dash, y_dash])

            A[2 * i] = row1
            A[2 * i + 1] = row2

        # H obtained through last vector of Singular Value Decomposition (SVD) of A matrix. SVD P = USV'
        U, S, V = np.linalg.svd(A, full_matrices=True)
        V = np.transpose(V)
        
        H = V[:, -1].reshape((3, 3))
        H = H / H[2, 2]
        
        return H
    

    def getInitKMatrix(self, H_matrices):
        """
        Estimates the camera intrinsic matrix based on the given array of homography matrices.

        Parameters
        ----------
        H_matrices : array-like
            An array of homography matrices.
        
        Results
        -------
        K_matrix : array-like
            A 3 x 3 matrix which is the camera intrisic matrix.
        """
        print('\nEstimating the camera intrinsic matrix...')
        # get the intial estimation of B matrix
        B_matrix = getBMatrix(H_matrices)
        B11, B12, B22, B13, B23, B33 = B_matrix

        # get the intrinsic parameters from the B matrix
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
        lamda = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = math.sqrt(lamda / B11)
        beta = math.sqrt(lamda * B11 / (B11 * B22 - B12 ** 2))
        gamma = -B12 * (alpha ** 2) * beta / lamda
        u0 = gamma * v0 / beta - B13 * (alpha ** 2) / lamda

        K_matrix = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

        return K_matrix

    
    def getInit_k(self):
        """
        Gets the initial estimate of camera's minimum distortion.

        Results
        -------
        k : array-like
            Array containing the coefficients of radial distortion.
        """
        k = np.array([0, 0])
        return k


    def getExtrinsicParameters(self, K, H_matrices):
        """
        Estimates the camera extrinsic parameters.

        Parameters
        ----------
        K : array-like
            A 3 x 3 camera intrinsic matrix.
        H_matrices : array-like
            An array of homography matrices.

        Results
        -------
        RT_array : array-like
            Array containing all the rotation matrices along with corresponding translation.
        """
        RT_list = []

        print('\nEstimating camera extrinsic parameters...\n')
        
        for H in H_matrices:
            lamda = 1 / np.linalg.norm((np.dot(np.linalg.inv(K), H[:, 0])), ord=2)

            r1 = lamda * np.dot(np.linalg.inv(K), H[:, 0])
            r2 = lamda * np.dot(np.linalg.inv(K), H[:, 1])
            r3 = np.cross(r1, r2)
            t = lamda * np.dot(np.linalg.inv(K), H[:, 2])
            
            R = np.column_stack((r1, r2, r3, t))
            RT_list.append(R)        
        
        RT_array = np.array(RT_list)

        return RT_array

    
    def reprojectPointsAndComputeError(self, K, k, RT_array, image_points, world_points):
        """
        Computes the reprojected points and corresponding error.

        Parameters
        ----------
        initial_parameters : dict
            Dictionary containing initial estimates of all parameters. 
        world_points : array-like
            An array containing all the world points in an image.
        image_points : array-like
            An array containing all the image points in an image.

        Results
        -------
        avg_err : float
            The average of allthe errors of reprojection.
        reprojected_points: array-like
            Array containing the reprojected points.
        """
        err_list = []
        reprojected_points = []

        alpha = K[0, 0]
        beta = K[1, 1]
        gamma = K[0, 1]
        u0 = K[0, 2]
        v0 = K[1, 2]

        k1 = k[0]
        k2 = k[1]
        
        for i in range(len(world_points)):

            # for 3D world points
            RT = RT_array[i]

            # for 2D world points
            RT3 = np.transpose(np.array([RT[:, 0], RT[:, 1], RT[:, 3]]).reshape(3, 3))
            ART = np.dot(K, RT3)

            err = 0
            reprojected_points_per_image = []

            for j, world_point in enumerate(world_points[i]):

                world_point_2d = np.array([world_point[0], world_point[1], 1])
                world_point_2d = np.reshape(world_point_2d, (3, 1))
                world_point_3d = np.array([world_point[0], world_point[1], 0, 1])
                world_point_3d = np.reshape(world_point_3d, (4, 1))

                xy = np.dot(RT, world_point_3d)
                x = xy[0] / xy[2]
                y = xy[1] / xy[2]

                mij = image_points[i, j].reshape((2, 1))
                mij = np.append(mij, 1)
                mij = mij.reshape((3, 1))

                m_hat = np.dot(ART, world_point_2d)
                m_hat = m_hat / m_hat[2]
                u, v = m_hat[0], m_hat[1]

                u_hat = u + (u - u0) * (k1 * (x ** 2 + y ** 2)  + k2 * (x ** 2 + y ** 2) ** 2)
                v_hat = v + (v - v0) * (k1 * (x ** 2 + y ** 2)  + k2 * (x ** 2 + y ** 2) ** 2)
                reprojected_points_per_image.append([u_hat, v_hat])

                m_hat = np.array([u_hat[0], v_hat[0], 1])
                m_hat = np.reshape(m_hat, (3, 1))

                err += np.linalg.norm((mij - m_hat), ord=2)
            
            err_list.append(err)
            reprojected_points.append(reprojected_points_per_image)

        err_array = np.array(err_list)
        avg_err = err_array.sum() / (world_points.shape[0] * world_points.shape[1])

        reprojected_points = np.array(reprojected_points)
        return avg_err, reprojected_points
            


def loss_func(x0, RT_array, image_points, world_points):
    """
    Loss function for non-linear optimization.

    Parameters
    ----------
    RT_array : array-like
        Array containing all the rotation matrices along with corresponding translation.
    world_points : array-like
        An array containing all the world points in an image.
    image_points : array-like
        An array containing all the image points in an image.
    
    Results
    -------
    err_array : array-like
        An array containing all the errors of reprojection.
    """
    err_list = []

    alpha, gamma, beta, u0, v0, k1, k2 = x0
    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    k = np.array([k1, k2])


    for i, worldPts in enumerate(world_points):

        RT = RT_array[i]
        RT3 = np.transpose(np.array([RT[:, 0], RT[:, 1], RT[:, 3]]).reshape(3, 3))
        ARt = np.dot(K, RT3)

        err = 0

        for j, world_point in enumerate(worldPts):
            world_point_2d = np.array([world_point[0], world_point[1], 1])
            world_point_2d = np.reshape(world_point_2d, (3, 1))
            world_point_3d = np.array([world_point[0], world_point[1], 0, 1])
            world_point_3d = np.reshape(world_point_3d, (4, 1))

            xy = np.dot(RT, world_point_3d)
            x = xy[0] / xy[2]
            y = xy[1] / xy[2]

            mij = image_points[i, j].reshape((2, 1))
            mij = np.append(mij, 1)
            mij = mij.reshape((3, 1))

            m_hat = np.dot(ARt, world_point_2d)
            m_hat = m_hat / m_hat[2]
            u, v = m_hat[0], m_hat[1]

            u_hat = u + (u - u0) * (k1 * (x ** 2 + y ** 2)  + k2 * (x ** 2 + y ** 2) ** 2)
            v_hat = v + (v - v0) * (k1 * (x ** 2 + y ** 2)  + k2 * (x ** 2 + y ** 2) ** 2)

            m = np.array([u_hat[0], v_hat[0], 1])
            m = np.reshape(m, (3, 1))

            err += np.linalg.norm((mij - m), ord=2)
        
        err_list.append(err)
        # err_array = np.array(err_list)
    
    return np.array(err_list)



def main():
    
    parser = argparse.ArgumentParser(description='\nImplementation of Zhang\'s Camera Calibration Method',
        usage='use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--images_path', default='./Code/Calibration_Imgs',
        help= textwrap.dedent('''\
        The path of folder containing the input images for calibration.

        default: ./Code/Calibration_Imgs
        '''))
    parser.add_argument('--rows', default=9, type=int,
        help= textwrap.dedent('''\
        Number of rows in chessboard pattern. 
        
        default: 9
        '''))
    parser.add_argument("--columns", type=int, default=6,
        help= textwrap.dedent('''\
        Number of columns in chessboard pattern. 
        
        default: 6
        '''))
    parser.add_argument("--square_size", type=float, default=21.5,
        help= textwrap.dedent('''\
        The length of edge of squares in the chessboard pattern. 
        
        default: 21.5
        '''))
    

    images_path = './Code/Calibration_Imgs'
    args = parser.parse_args()
    images_path = str(args.images_path)
    pattern_rows = args.rows
    pattern_cols = args.columns
    square_size = args.square_size

    print('\n\n')
    images = readImages(images_path)
    imgs = images.copy()

    calibrator = AutoCalibrator(imgs, (pattern_rows, pattern_cols), square_size)

    # get the world points (points on checkerboard) and the image points from the input images
    image_points, world_points = calibrator.getChessBoardCorners()

    # get the initial estimates of intrinsic and extrinsic parameters
    K_matrix_initial, k_initial, RT_array = calibrator.getInitialParameterEstimates(image_points, world_points)

    error, _ = calibrator.reprojectPointsAndComputeError(K_matrix_initial, k_initial, RT_array, image_points, world_points)
    print('\nError before optimization: ', error)

    print('\nNon-linear optimization initiating...')

    alpha = K_matrix_initial[0, 0]
    beta = K_matrix_initial[1, 1]
    gamma = K_matrix_initial[0, 1]
    u0 = K_matrix_initial[0, 2]
    v0 = K_matrix_initial[1, 2]

    k1 = k_initial[0]
    k2 = k_initial[1]
    x0 = np.array([alpha, gamma, beta, u0, v0, k1, k2])


    res = optimize.least_squares(fun=loss_func, x0=x0, method="lm", args=[RT_array, image_points, world_points])
    print('\nOptimization Complete!')
    alpha_new, gamma_new, beta_new, u0_new, v0_new, k1_new, k2_new = res.x
    K_new = np.array([[alpha_new, gamma_new, u0_new], [0, beta_new, v0_new], [0, 0, 1]])
    k_new = np.array([k1_new, k2_new])
    print('\nNew K: \n', K_new)
    print('\nNew k: \n', k_new)

    error, reprojected_points = calibrator.reprojectPointsAndComputeError(K_new, k_new, RT_array, image_points, world_points)
    print('\nError after optimization: ', error, '\n')

    # visualizing the reprojection of points on the images
    K = np.array(K_new, np.float32).reshape(3,3)
    D = np.array([k_new[0],k_new[1], 0, 0] , np.float32)

    for i,image_points in enumerate(reprojected_points):
        image = cv2.undistort(images[i], K, D)
        for point in image_points:
            x = int(point[0])
            y = int(point[1])
            image = cv2.circle(image, (x, y), 5, (0, 0, 255), 3)
        
        if not os.path.exists('.Code/Results'):
            os.mkdir('./Code/Results')
        cv2.imwrite(f"./Code/Results/reproj{i+1}.png", image)

    cv2.destroyAllWindows()

   
 
if __name__ == '__main__':
    main()