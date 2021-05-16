import tensorflow as tf

fw = open("node_names.log", 'w')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./openpose_hand.ckpt.meta', clear_devices=True)
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    node_list = [n.name for n in graph_def.node]
    for node in node_list:
        print("node_name", node)
        fw.write("node_name:{}\n".format(node))

fw.close()
