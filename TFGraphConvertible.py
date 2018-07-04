import tensorflow as tf

class TFGraphConvertible():
    def to_graph(self, fields):
        def __inner__(root, k, var):
            if self.is_tf_convertible(var):
                return var.name

            if isinstance(var, list):
                if k not in root.keys():
                    root[k] = []
                for i, v in enumerate(var):
                    out = __inner__(root[k], i, v)
                    if out is not None:
                        root[k].append(out)
                if len(root[k]) == 0:
                    del root[k]

            if isinstance(var, dict):
                if k not in root.keys():
                    root[k] = {}
                for i, v in var.items():
                    out = __inner__(root[k], i, v)
                    if out is not None:
                        root[k][i] = out
                if len(root[k].keys()) == 0:
                        del root[k]
            return None

        graph = {}

        for k, v in fields.items():
            out = __inner__(graph, k, v)
            if out is not None:
                graph[k] = out

        return graph

    def from_graph(self, graph, tf_graph):
        def __inner__(root, vars):
            for k, var in vars:
                if isinstance(var, list):
                    if isinstance(root, object) or isinstance(root, dict):
                        setattr(root, k, var)
                    __inner__(var, enumerate(var))
                elif isinstance(var, dict):
                    if isinstance(root, object) or isinstance(root, dict):
                        setattr(root, k, var)
                    __inner__(var, var.items())
                else:
                    try:
                        tf_var = tf_graph.get_tensor_by_name(var)
                    except ValueError:
                        tf_var = tf_graph.get_operation_by_name(var)
                        pass
                    if isinstance(root, list) or isinstance(root, dict):
                        root[k] = tf_var
                    elif isinstance(root, object):
                        setattr(root, k, tf_var)
            return

        __inner__(self, graph.items())

    @staticmethod
    def is_tf_convertible(x):
        return isinstance(x, (tf.Tensor,
                              tf.Variable,
                              tf.SparseTensor,
                              tf.Operation,
                              ))