from onnx_tf.backend import prepareimport onnxTF_PATH = "./data/MNIST.pb"ONNX_PATH = './data/MNIST.onnx'onnx_model = onnx.load_model(ONNX_PATH)tf_rep = prepare(onnx_model)tf_rep.export_graph(TF_PATH)