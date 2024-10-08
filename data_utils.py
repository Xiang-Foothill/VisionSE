from typing import Tuple
import numpy as np
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import os

def parse_barc_data(dataset_path = "ParaDriveLocalComparison_Oct1_enc_0.npz") -> Tuple[np.ndarray, np.ndarray]:
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
    return images, np.linalg.norm(states[:, :2], axis=1)  # velocity magnitude 

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

# if __name__ == '__main__':
#     parse_barc_data(Path.cwd() / 'ParaDriveLocalComparison_Sep7_0.npz')
