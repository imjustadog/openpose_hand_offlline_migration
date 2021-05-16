#from keras.models import load_model
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format

import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
import argparse
import os

def ConvertSavedModelToPB(input_model, output_node_names, freeze):
  with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], input_model)
    output_graph = sess.graph.as_graph_def()
    if freeze == False:
      output_graph = sess.graph.as_graph_def()
    else:
      output_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                       sess.graph_def, output_node_names)
  return graph_util.extract_sub_graph(output_graph, output_node_names)


def ConvertPBTXTToPB(input_model):
  with tf.Session() as sess:
    with tf.gfile.FastGFile(input_model, 'rb') as f:
      output_graph = text_format.Merge(f.read(), sess.graph.as_graph_def())
  return output_graph 


#def ConvertKerasHDF5ModelToPB(input_model):
#  model = load_model(input_model)
#  sess = KTF.get_session()
#  output_node_names = [node.op.name for node in model.outputs]
#  if freeze == False:
#    output_graph = sess.graph.as_graph_def()
#  else:
#    output_graph = tf.graph_util.convert_variables_to_constants(sess,
#            sess.graph_def, output_node_names)
#  return graph_util.extract_sub_graph(output_graph, output_node_names)


def ConvertMetaGraphModelToPB(input_model, ckpt, output_node_names, freeze):
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(input_model)
    if freeze == False:
      output_graph = sess.graph.as_graph_def()
    else:
      saver.restore(sess, tf.train.latest_checkpoint(ckpt))
      output_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                       sess.graph_def, output_node_names)
  return graph_util.extract_sub_graph(output_graph, output_node_names)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", help="Input model name", required=True)
  parser.add_argument("--input_model", help="Input model path", required=True)
  parser.add_argument("--output_model_dir", help="Output directory to store frozen PB model", required=True)
  parser.add_argument("--output_node_names", help="Graph output node names", nargs="*", required=True)
  parser.add_argument("--ckpt", help="Checkpoint data path for meta graph")
  parser.add_argument("--freeze", help="Whether need to freeze weights into PB model", action="store_true")
  args = parser.parse_args()

  model_name = args.model_name
  input_model = args.input_model
  output_model_dir = args.output_model_dir
  freeze = args.freeze
  output_node_names = args.output_node_names

  # remove ending "/" of directory
  input_model.rstrip("/")
  root, model_ext = os.path.splitext(input_model)

  if os.path.isdir(input_model) and len(model_ext) == 0:
    #SavedModel format
    output_graph = ConvertSavedModelToPB(input_model, output_node_names, freeze)
#  elif model_ext.lower() in [".h5"]:
#    #Keras HDF5 format
#    output_graph = ConvertKerasHDF5ModelToPB(input_model, freeze)
  elif model_ext.lower() in [".meta"]:
    #Tensorflow meta graph with ckpt data
    ckpt = args.ckpt
    output_node_names = args.output_node_names
    output_graph = ConvertMetaGraphModelToPB(input_model, ckpt, output_node_names, freeze)
  else:
    raise ValueError("Unsupport model format {}".format(input_model))

  # Save PB models
  with tf.gfile.GFile(os.path.join(output_model_dir, model_name + ".pb"), "wb") as f:
         f.write(output_graph.SerializeToString())
