原github仓库：
https://github.com/ethereon/caffe-tensorflow


.
├── convert.py
├── examples
│   ├── imagenet
│   │   ├── classify.py
│   │   ├── dataset.py
│   │   ├── imagenet-classes.txt
│   │   ├── models
│   │   │   ├── alexnet.py
│   │   │   ├── caffenet.py
│   │   │   ├── googlenet.py
│   │   │   ├── helper.py
│   │   │   ├── __init__.py
│   │   │   ├── nin.py
│   │   │   ├── resnet.py
│   │   │   └── vgg.py
│   │   ├── README.md
│   │   └── validate.py
│   └── mnist
│       ├── finetune_mnist.py
│       ├── lenet_iter_10000.caffemodel
│       ├── lenet.prototxt
│       └── README.md
├── kaffe
│   ├── caffe
│   │   ├── caffepb.py
│   │   ├── caffepb.pyc
│   │   ├── __init__.py
│   │   ├── __init__.pyc
│   │   ├── resolver.py
│   │   └── resolver.pyc
│   ├── errors.py
│   ├── errors.pyc
│   ├── graph.py
│   ├── graph.pyc
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── layers.py
│   ├── layers.pyc
│   ├── shapes.py
│   ├── shapes.pyc
│   ├── tensorflow
│   │   ├── __init__.py
│   │   ├── __init__.pyc
│   │   ├── network.py
│   │   ├── network.pyc
│   │   ├── transformer.py
│   │   └── transformer.pyc
│   ├── transformers.py
│   └── transformers.pyc
├── LICENSE.md
├── openpose_hand.npy
├── openpose_hand.py
├── pose_deploy.prototxt
├── pose_iter_102000.caffemodel
└── README.md




pip2 install tensorflow==1.0.0
pip2 install protobuf==3.1
export PYTHONPATH='/home/adoge/openpose/transfer/caffe-tensorflow'

python2 convert.py --def_path ./pose_deploy.prototxt --caffemodel ./pose_iter_102000.caffemodel --data-output-path ./openpose_hand.npy --code-output-path ./openpose_hand.py
