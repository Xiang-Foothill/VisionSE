from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
import os
BARC_H = 0.123 # the height of the camera from the horizontal graound 
BARC_F = 605.5 # focal length in terms of pixels - [pixels]
BARC_T = 0.1

"""DATA prase functions below
the format of data in the data pacakges collected by the carla gym script:
each data package is saved as a pkl file,
Opening the file, what you will find is a dictionary with three keys: images, states, and imu (in the format of string)

data["images"]: a length N array, where each element is a W*H size-image
data["states"]: a length N array, where each element is an inner nested array of the form [V_x, V_y, w], which is [velocity at the x-direction, velocity at the y-direction, and angular velocity]
data["imu"]: a length N array, where each element is an inner nested array of the form [a_long, w]. a_long is the measured linear acceleration in the longitudinal direction by the imu, and w is the measured angular velocity by the imu

All three length-N arrays are time-series with the same time step, i.e. the ith image is collected as the same time as the ith state, and ith imu measurement.
"""


def full_parse(dataset_path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse the data needed for the optical flow algorithm

    the order of parameters returned:
    images, V_tran, V_long, w, f, h, deltaT"""
    cur_path = os.getcwd()
    SE_root = os.path.dirname(cur_path)
    dataset_path = SE_root + "/VideoSet/" + dataset_path
    data = np.load(dataset_path, allow_pickle=True)
    images, states = data['images'], data['states']
    images = np.asarray(images)
    states = np.asarray(states)
    images = to_cvChannels(images)

    if "F" in data:
        return images, np.linalg.norm(states[:, :2], axis=1), states[:, 2], data["F"], data["sensor_height"], data["T"]  # velocity magnitude 
    else:
        return images, np.linalg.norm(states[:, :2], axis=1), states[:, 1], states[:, 2], BARC_F, BARC_H, BARC_T

def imu_parse(dataset_path):
    """this function only parses data from the imu sensor"""
    cur_path = os.getcwd()
    SE_root = os.path.dirname(cur_path)
    dataset_path = SE_root + "/VideoSet/" + dataset_path
    data = np.load(dataset_path, allow_pickle=True)

    imu_data = data["imu"]
    imu_data = np.asarray(imu_data)
    return imu_data

def parse_barc_data(dataset_path, Omega_exist = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    This prase function is not reliable, PLEASE DO NOT USE IT.
    @param dataset_path: Path to the dataset npz file. 
    @param Omega_exist: if the values of angular velocities should be returned or not, if they need to be returned, return in the form: [Images, real_Vs, real_Omegas, F, h, T]
    @return: (image, velocity)
    """
    cur_path = os.getcwd()
    SE_root = os.path.dirname(cur_path)
    dataset_path = SE_root + "/VideoSet/" + dataset_path
    data = np.load(dataset_path, allow_pickle=True)

    # {'images': (N, C, H, W), 
    # 'sensors': (N, O), 
    # 'states': (N, S),  # [v_long, v_tran, w_psi, x, y, psi] 
    # 'actions': (N, A),  # expert action
    # 'agent_actions': (N, A), 
    # 'collection_actions': (N, A), 
    # 'rews': (N, ), 
    # 'dones': (N, )}

    images, states = data['images'], data['states']

    images = np.asarray(images)
    states = np.asarray(states)
    images = to_cvChannels(images)

    if Omega_exist:
        if "F" in data:
            return images, np.linalg.norm(states[:, :2], axis=1), states[:, 2], data["F"], data["sensor_height"], data["T"]  # velocity magnitude 
        else:
            return images, np.linalg.norm(states[:, :2], axis=1), states[:, 2], BARC_F, BARC_H, BARC_T
    else:
        if "F" in data:
            return images, np.linalg.norm(states[:, :2], axis=1), data["F"], data["sensor_height"], data["T"]  # velocity magnitude 
        else:
            return images, np.linalg.norm(states[:, :2], axis=1), BARC_F, BARC_H, BARC_T

def to_cvChannels(img):
    """convert the RGB image from the numpy format to the cv2 format
    cv2 format: [N, height, width, channels]
    numpy format:[N, channels, height, wdith]"""
    if img.shape[3] == 3: # first check it is already in the form of cv2. If it is return it directly
        return np.uint8(img)
    
    N, h, w = img.shape[0], img.shape[2], img.shape[3]
    cv2_img = np.zeros(shape = (N, h, w, 3))
    cv2_img[:, :, :, 0] = img[:, 2, :, :]
    cv2_img[:, :, :, 1] = img[:, 1, :, :]
    cv2_img[:, :, :, 2] = img[:, 0, :, :]

    return np.uint8(cv2_img)

def to_npChannels(img):
    """convert a RGB image from the cv2 format to the numpy format
    cv2 format: [height, width, channels]
    numpy format:[channels, height, wdith]"""
    h, w = img.shape[0], img.shape[1]
    res = np.zeros(shape = (3, h, w))
    res[0, :, :] = img[:, :, 0]
    res[1, :, :] = img[:, :, 1]
    res[2, :, :] = img[:, :, 2]

    return np.uint8(res)

def BGRA2RGB(img):
    """format of BGRA: [width, height, channels = 4]
    formate of RGB: [channels = 3, width, height]"""
    img = img[:, :, :4]
    w, h = img.shape[0], img.shape[1]
    np_rgb = np.zeros(shape = (w, h, 3))

    np_rgb[:, :, 0] = img[:, :, 2]
    np_rgb[:, :, 1] = img[:, :, 1]
    np_rgb[:, :, 2] = img[:, :, 0]
    return np_rgb

def BGR2RGB(img):
    """format of BGR: [height, width, channels]
    formate of RGB: [height, width, channels]"""
    w, h = img.shape[0], img.shape[1]
    np_rgb = np.zeros(shape = (w, h, 3))
    np_rgb[:, :, 0] = img[:, :, 2]
    np_rgb[:, :, 1] = img[:, :, 1]
    np_rgb[:, :, 2] = img[:, :, 0]

    return np_rgb
def random_image_test(images):
    im = images[np.random.randint(low = 0, high = images.shape[0])]
    imgplot = plt.imshow(im[1, :, :])
    plt.show()
