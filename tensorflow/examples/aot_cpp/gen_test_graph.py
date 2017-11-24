"""
Create test graph for export
"""
import tensorflow as tf

from pathlib import Path

tf_root_dir = Path.home() / 'CodeBase' / 'dbml-root-workspace' / 'tnsrphlw'
example_root_dir = tf_root_dir / 'tensorflow/tensorflow/examples/aot_cpp'

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    A = tf.placeholder(tf.float64, shape=(100, 200), name='A')
    B = tf.placeholder(tf.float64, shape=(200, 100), name='B')
    C = tf.matmul(A, B, name='C')


# write_visualization_html(graph)

Av = np.random.randn(100, 200)
Bv = np.random.randn(200, 100)

with tf.Session(graph=graph) as sess:
    vals = C.eval({A: Av, B: Bv})

assert np.allclose(Av @ Bv, vals)

with tf.gfile.Open(str(example_root_dir / 'test_graph_tfmatmul.pb'), 'wb') as fout:
    fout.write(graph.as_graph_def().SerializeToString())
