import torch
import torch.nn as nn
import collections
from copy import deepcopy
import sys
sys.path.append('./')
from graphs.base_graph import BIGGraph, NodeType

class DenseNetGraph(BIGGraph):
    def __init__(self, model, layer_name='features', head_name='classifier'):
        super().__init__(model)
        self.layer_name = layer_name
        self.head_name = head_name
        self.outputs = collections.OrderedDict()  # To store outputs

    def add_dense_block_nodes(self, name_prefix, input_node):
        dense_block = self.get_module(name_prefix)
        previous_output_nodes = [input_node]

        for i, layer in enumerate(dense_block.children()):
            layer_name = f'{name_prefix}.denselayer{i+1}'

            output_node = self.add_nodes_from_sequence(
                name_prefix=layer_name,
                list_of_names=['norm1', 'conv1', 'norm2', 'conv2', NodeType.CONCAT],
                input_node=previous_output_nodes[-1],
            )
            if len(previous_output_nodes) > 1:
                for prev_node in previous_output_nodes:
                    self.add_directed_edge(prev_node, output_node)
            previous_output_nodes.append(output_node)

        return previous_output_nodes[-1]

    def add_transition_layer_nodes(self, name_prefix, input_node):
        transition_layer = self.get_module(name_prefix)
        return self.add_nodes_from_sequence(name_prefix, ['norm', 'conv', NodeType.PREFIX], input_node)

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        input_node = self.add_nodes_from_sequence('', ['features.conv0', 'features.norm0'], input_node, sep='')

        dl = 1
        tl = 1
        for i, block in enumerate(self.get_module(self.layer_name)):
            block_class = block.__class__.__name__
            if block_class == '_DenseBlock':
                input_node = self.add_dense_block_nodes(f'{self.layer_name}.denseblock{dl}', input_node)
                dl += 1
                print('dense')
            elif block_class == '_Transition':
                input_node = self.add_transition_layer_nodes(f'{self.layer_name}.transition{tl}', input_node)
                tl += 1
                print('transition')

        input_node = self.add_nodes_from_sequence('', ['features.norm5', 'adaptive_avgpool2d', self.head_name, NodeType.OUTPUT], input_node, sep='')

        return self
    
    
    def forward(self, x):
        self.outputs.clear()

        def hook_fn(name, module):
            def hook(module, input, output):
                print(name, output.shape)
                self.outputs[name] = output
            return hook

        # Register hooks
        for name, module in self.model.named_modules():
            module.register_forward_hook(hook_fn(name, module))

        # Forward pass through the original model
        x = self.get_module('features')(x)
        x = self.get_module(self.head_name)(x)

        return x



import torch
import torchvision.models as models

def test_graph_model():
    # Define or load your DenseNet model
    original_model = models.densenet121(pretrained=False).eval()


    # Set the model to evaluation mode
    original_model.eval()

    # Create and graphify the DenseNetGraph
    graph = DenseNetGraph(original_model)
    graph.graphify()

    # Create a dummy input tensor
    input_tensor = torch.randn(1, 3, 224, 224)  # Adjust size as per your model

    # Perform a forward pass through the original model
    with torch.no_grad():
        _ = original_model(input_tensor)  # Pass through the original model
    print(_.shape)
    print('doneeeeeeeeeeeeeeeee')
    # Perform a forward pass through the graph-based model
    with torch.no_grad():
        print('yes')
        _ = graph.forward(input_tensor)  # Pass through the graph-based model
        print('who')
    # Print the shapes of intermediate outputs
    for name, output in graph.outputs.items():
        print(f'Layer: {name}, Output shape: {output.shape}')

# Run the test
test_graph_model()