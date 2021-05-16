原github仓库：
https://github.com/alesolano/npy2ckpt



.
├── checkpoint
├── models
│   ├── imagenet-classes.txt
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── openpose_hand.npy
│   ├── openpose_hand.py
│   ├── openpose_hand.pyc
│   └── __pycache__
│       ├── __init__.cpython-36.pyc
│       └── openpose_hand.cpython-36.pyc
├── network.py
├── network.pyc
├── npy2ckpt_GoogleNet.py
├── npy2ckpt_OpenPose.py
├── npy2ckpt.py
├── openpose_hand.ckpt.data-00000-of-00001
├── openpose_hand.ckpt.index
├── openpose_hand.ckpt.meta
├── __pycache__
│   └── network.cpython-36.pyc
├── README.md
├── test_GoogleNet.py
└── tests
    ├── daisy_test.jpg
    └── dog.jpg




models文件夹存放__init__.py(空的即可）, openpose_hand.py, openpose_hand.npy

openpose_hand.py前两行改成：
from network import Network
class OpenPoseNet(Network):

npy2ckpt_OpenPose.py 第三行改成：
from models.openpose_hand import OpenPoseNet

python2 npy2ckpt_OpenPose.py --model_path ./models/openpose_hand.npy --output_path ./openpose_hand.ckpt
