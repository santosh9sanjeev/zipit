import torch
import networkx as nx
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class FeatureReshapeHandler:
    """Instructions to reshape layer intermediates for alignment metric computation."""

    def handle_conv2d(self, x):
        # reshapes conv2d representation from [B, C, H, W] to [C, -1]
        B, C, H, W = x.shape
        return x.permute(1, 0, 2, 3).reshape(C, -1)

    def handle_linear(self, x):
        # x is shape [..., C]. Want [C, -1]
        x = x.flatten(0, len(x.shape) - 2).transpose(1, 0).contiguous()
        return x

    def __init__(self, class_name, info):
        self.handler = {
            'BatchNorm2d': self.handle_conv2d,
            'LayerNorm': self.handle_linear,
            'Conv2d': self.handle_conv2d,
            'Linear': self.handle_linear,
            'GELU': self.handle_linear,
            'AdaptiveAvgPool2d': self.handle_conv2d,
            'LeakyReLU': self.handle_conv2d,
            'Tanh': self.handle_conv2d,
            'MaxPool2d': self.handle_conv2d,
            'AvgPool2d': self.handle_conv2d,
            'SpaceInterceptor': self.handle_conv2d,
            'Identity': self.handle_linear,
        }[class_name]
        self.info = info
    def reshape(self, x):
        x = self.handler(x)

        # Handle modules that we only want a piece of
        if self.info['chunk'] is not None:
            idx, num_chunks = self.info['chunk']
            x = x.chunk(num_chunks, dim=0)[idx]

        return x
    
class NodeType(Enum):
    MODULE = 0          # node is torch module
    PREFIX = 1          # node is a PREFIX (i.e., we want to hook inputs to child node)
    POSTFIX = 2         # node is a POSTFIX  (i.e., we want to hook outputs to parent node)
    SUM = 3             # node is a SUM (e.g., point where residual connections are connected - added)
    CONCAT = 4          # node is a CONCATENATION (e.g., point where residual connections are concatenated)
    INPUT = 5           # node is an INPUT (graph starting point)
    OUTPUT = 6          # node is an OUTPUT (graph output point)
    EMBEDDING = 7       # node is an embedding module (these can only be merged)


class BIGGraph(ABC):
    def __init__(self, model):
        """Initialize DAG of computational flow for a model."""
        self.reset_graph()
        self.named_modules = dict(model.named_modules())
        self.named_params = dict(model.named_parameters())
        self.model = model
        self.intermediates = {}
        self.hooks = []

        # working info about nodes for the merging algorithms
        # clear after use!
        self.working_info = {}

        self.unmerged = set()
        self.merged = set()

    def reset_graph(self):
        """Create New Graph."""
        self.G = nx.DiGraph()

    def preds(self, node):
        """Get predecessors from a node (layer)."""
        return list(self.G.pred[node])

    def succs(self, node):
        """Get successors from a node (layer)."""
        return list(self.G.succ[node])

    def get_node_info(self, node_name):
        """Get attribute dict from node."""
        return self.G.nodes[node_name]

    def get_module_from_node(self, node_name):
        """Get pytorch module associated with node."""
        info = self.get_node_info(node_name)
        if info['type'] == NodeType.MODULE:
            return self.named_modules[info["layer"]]
        else:
            raise ValueError(f"Tried to get module from {node_name} of type {info['type']}.")

    def get_module(self, module_name):
        """Get module parameters."""
        return self.named_modules[module_name]

    def get_parameter(self, param_name):
        """Get parameter from name."""
        return self.named_params[param_name]

    def get_node_str(self, node_name):
        """Get node type name."""
        info = self.get_node_info(node_name)
        if info['type'] == NodeType.MODULE:
            return self.get_module_from_node(node_name).__class__.__name__
        else:
            return info['type'].name

    def create_node_name(self):
        """A robust id generator"""
        return len(self.G)

    def create_node(self,
                    node_name=None,
                    layer_name=None,
                    param_name=None,
                    node_type=NodeType.MODULE,
                    chunk=None,
                    special_merge=None,
                    input_shape=None,
                    output_shape=None):
        """Create node to be added to graph. All arguments are optional, but specify different kinds of node properties."""
        if node_name is None:
            node_name = self.create_node_name()
        self.G.add_nodes_from([(node_name, {
            'layer': layer_name,
            'type': node_type,
            'param': param_name,
            'chunk': chunk,
            'special_merge': special_merge,
            'input_shape': input_shape,
            'output_shape': output_shape
        })])
        return node_name

    def add_directed_edge(self, source, target, **kwargs):
        """Add an edge from source node to target node."""
        self.G.add_edge(source, target, **kwargs)

    def add_nodes_from_sequence(self, name_prefix, list_of_names, input_node, sep='.'):
        """
        Add multiple nodes in sequence by creating them and adding edges between each.
        Args:
        - name_prefix: Least common ancestor module name string all nodes share. Usually this is the name of a nn.Sequential layer
        - list_of_names: list of module names. Can be the ordered module names in an nn.Sequential.
        - input_node: source node the sequence is attached to.
        Returns:
        - output sequence node.
        """
        source_node = input_node
        for name in list_of_names:
            if isinstance(name, str):
                temp_node = self.create_node(layer_name=name_prefix + f'{sep}{name}')
            else:
                temp_node = self.create_node(node_type=name)
            self.add_directed_edge(source_node, temp_node)
            source_node = temp_node
        return source_node

    def print_prefix(self):
        """Print (POST/PRE)FIX node inputs and outputs."""
        for node in self.G:
            info = self.get_node_info(node)
            if info['type'] in (NodeType.PREFIX, NodeType.POSTFIX):
                print(f'{node:3} in={len(self.preds(node))}, out={len(self.succs(node))}')

    def draw(self, nodes=None, save_path=None):
        """
        Visualize DAG. By default all nodes are colored gray, but if parts of module have already been transformed, they will be colored according to the kinds of transformations applied on the nodes.
        Color Rubric:
        - Gray: Not merged or unmerged  (output space is not aligned        and input space is not  aligned)
        - Blue: merged but not unmerged (output space is     aligned,       but input space is not  aligned)
        - Red: Not merged but unmerged  (output space is not aligned,       but input space is      aligned)
        - Pink: Merged and Unmerged     (output space is     aligned,       and input space is      aligned)

        Args:
        - nodes (optional): list of indices of nodes to visualize. If None, all nodes will be drawn.
        - save_path (optional): path in which to save graph.
        """
        G = self.G
        if nodes is not None:
            G = nx.subgraph(G, list(nodes))

        labels = {i: f'[{i}] ' + self.get_node_str(i) for i in G}
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        node_size = [len(labels[i])**2 * 60 for i in G]

        colors = {
            (False, False): (180, 181, 184),
            (True, False): (41, 94, 255),
            (False, True): (255, 41, 91),
            (True, True): (223, 41, 255),
        }

        for k, v in colors.items():
            colors[k] = tuple(map(lambda x: x / 255., v))

        node_color = [colors[(node in self.merged, node in self.unmerged)] for node in G]
        plt.figure(figsize=(120, 160))
        nx.draw_networkx(G, pos=pos, labels=labels, node_size=node_size, node_color=node_color)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def add_hooks(self, device=0):
        """Propagate PREFIX, POSTFIX, and capture Conv2d and BatchNorm2d shapes."""
        self.clear_hooks()

        for node in self.G:
            info = self.get_node_info(node)

            if info['type'] == NodeType.PREFIX:
                for succ_node in self.G.succ[node]:
                    succ_info = self.get_node_info(succ_node)
                    if succ_info['type'] == NodeType.MODULE:
                        def prehook(m, x, this_node=node, this_info=succ_info):
                            reshaped = FeatureReshapeHandler(m.__class__.__name__, this_info).reshape(x[0].detach().to(device))
                            self.intermediates[this_node] = reshaped
                            self.G.nodes[this_node]['shape'] = reshaped.shape  # Update shape info
                            return None

                        module = self.get_module(succ_info['layer'])
                        self.hooks.append(module.register_forward_pre_hook(prehook))
                        break
                else:
                    raise RuntimeError(f"PREFIX node {node} had no module to attach to.")

            elif info['type'] == NodeType.POSTFIX:
                for pred_node in self.G.pred[node]:
                    pred_info = self.get_node_info(pred_node)

                    if pred_info['type'] == NodeType.MODULE:
                        def posthook(m, x, y, this_node=node, this_info=pred_info):
                            reshaped = FeatureReshapeHandler(m.__class__.__name__, this_info).reshape(y.detach().to(device))
                            self.intermediates[this_node] = reshaped
                            self.G.nodes[this_node]['shape'] = reshaped.shape  # Update shape info
                            # info['output_shape'] = reshaped.shape
                            return None

                        module = self.get_module(pred_info['layer'])
                        self.hooks.append(module.register_forward_hook(posthook))
                        break

                    elif pred_info['type'] == NodeType.EMBEDDING:
                        def prehook(m, x, this_node=node, this_info=pred_info):
                            tensor = self.get_parameter(this_info['param']).data
                            tensor = tensor.flatten(0, len(x.shape)-2).transpose(1, 0).contiguous()
                            self.intermediates[this_node] = tensor
                            self.G.nodes[this_node]['shape'] = tensor.shape  # Update shape info
                            # info['output_shape'] = tensor.shape
                            return None

                        self.hooks.append(module.register_forward_pre_hook(self.model))

                else:
                    raise RuntimeError(f"POSTFIX node {node} had no module to attach to.")

            elif info['type'] == NodeType.MODULE and self.get_module_from_node(node).__class__.__name__ in ['Conv2d', 'BatchNorm2d']:
                # For Conv2d and BatchNorm2d layers, capture their output shape
                module = self.get_module_from_node(node)
                def shape_hook(m, x, y, this_node=node):
                    tensor_y =  y[0].view(y.size(1), -1)
                    self.G.nodes[this_node]['shape'] = tensor_y.shape  # Store the output shape
                    # info['output_shape'] = tensor_y.shape
                    return None

                self.hooks.append(module.register_forward_hook(shape_hook))

    
    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    
    def compute_intermediates(self, x):
        """ Computes all intermediates in a graph network. Takes in a torch tensor (e.g., a batch). """
        self.model = self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.intermediates = {}
            self.model(x)
            return self.intermediates
    
    
    @abstractmethod
    def graphify(self):
        """ 
        Abstract method. This function is implemented by your architecture graph file, and is what actually
        creates the graph for your model. 
        """
        return NotImplemented