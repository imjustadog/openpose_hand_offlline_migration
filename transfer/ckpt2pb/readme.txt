.
├── checkpoint
├── graph
├── hand.pb
├── openpose_hand.ckpt.data-00000-of-00001
├── openpose_hand.ckpt.index
├── openpose_hand.ckpt.meta
├── print_graph.py
├── print_node_name.py
├── readme.txt
└── x_to_pb.py




python3 x_to_pb.py --model_name hand --input_model ./openpose_hand.ckpt.meta --output_model_dir ./ --output_node_names Mconv7_stage6/BiasAdd(该节点名称通过看tensorboard和caffe原模型结合得到） --ckpt ./ --freeze
