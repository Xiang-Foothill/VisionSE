from typing import Tuple
import numpy as np
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import os
BARC_PATH = "ParaDriveLocalComparison_Oct1_enc_0.npz"
CARLA_PATH = "carlaData1.pkl"
BARC_H = 0.123 # the height of the camera from the horizontal graound 
BARC_F = 605.5 # focal length in terms of pixels - [pixels]

def parse_barc_data(dataset_path = CARLA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """
    
    @param dataset_path: Path to the dataset npz file. 
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

    # In "ParaDriveLocalComparison_Sep7_0.npz", N=1019, C=3, H=360, W=640, O=6, S=6, A=2. 
    # for key, value in data.items():
    #     print(f"{key}: {value.shape}")

    images, states = data['images'], data['states']

    # fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    # for i, idx in enumerate(np.random.permutation(images.shape[0])[:9]):
    #     ax = axes[i // 3][i % 3]
    #     ax.imshow(np.moveaxis(images[idx], [0, 1, 2], [2, 0, 1]))
    #     ax.set_title(f"#{idx}\t" + "$v_{long} = $" + f"{states[idx, 0]:.2f}")
    # plt.show()

    # return images, states[:, 0]  # longitudinal velocity mangnitude 
    images = to_cvChannels(images)
    if "F" in data:
        return images, np.linalg.norm(states[:, :2], axis=1), data["F"], data["sensor_height"]  # velocity magnitude 
    else:
        return images, np.linalg.norm(states[:, :2], axis=1), BARC_F, BARC_H
    
def to_cvChannels(img):
    """convert the RGB image from the numpy format to the cv2 format
    cv2 format: [N, height, width, channels]
    numpy format:[N, channels, height, wdith]"""
    N, h, w = img.shape[0], img.shape[2], img.shape[3]
    cv2_img = np.zeros(shape = (N, h, w, 3))
    cv2_img[:, :, :, 0] = img[:, 2, :, :]
    cv2_img[:, :, :, 1] = img[:, 1, :, :]
    cv2_img[:, :, :, 2] = img[:, 0, :, :]
    return np.uint8(cv2_img)

def BGRA2RGB(img):
    """format of BGRA: [width, height, channels = 4]
    formate of RGB: [channels = 3, width, height]"""
    img = img[:, :, :4]
    w, h = img.shape[0], img.shape[1]
    np_rgb = np.zeros(shape = (3, w, h))

    np_rgb[0, :, :] = img[:, :, 2]
    np_rgb[1, :, :] = img[:, :, 1]
    np_rgb[2, :, :] = img[:, :, 0]
    return np_rgb

def random_image_test(images):
    im = images[np.random.randint(low = 0, high = images.shape[0])]
    imgplot = plt.imshow(im[1, :, :])
    plt.show()

# if __name__ == '__main__':
#     parse_barc_data(Path.cwd() / 'ParaDriveLocalComparison_Sep7_0.npz')
