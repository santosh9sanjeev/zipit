import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.base_graph import BIGGraph, NodeType



class SimpleConcatNet(nn.Module):
    def __init__(self):
        super(SimpleConcatNet, self).__init__()
        # First branch
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Second branch
        self.conv3 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # After first concat
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # After second concat
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # First branch
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        
        # Second branch
        x2 = self.conv3(x)
        x2 = self.conv4(x2)
        
        # Concatenate the branches along the channel dimension
        x_cat1 = torch.cat((x1, x2), dim=1)  # Shape (B, 64, H, W)
        
        # Continue after concatenation
        x_cat1 = self.conv5(x_cat1)  # Shape (B, 32, H, W)
        
        # Another concatenation after more convolutions
        x_cat2 = torch.cat((x1, x_cat1), dim=1)  # Shape (B, 96, H, W)
        x_cat2 = self.conv6(x_cat2)  # Shape (B, 64, H, W)
        
        # Pooling and classification
        x_out = self.pool(x_cat2)
        x_out = torch.flatten(x_out, 1)
        x_out = self.fc(x_out)
        
        return x_out

class SimpleConcatNetGraph(BIGGraph):
    def __init__(self, model):
        super().__init__(model)
    
    def graphify(self):
        # Start by creating an input node
        input_node = self.create_node(node_type=NodeType.INPUT)
        
        # First branch (conv1 -> conv2), accessing layers correctly
        branch1_node = self.add_nodes_from_sequence(
            name_prefix='',  # You can modify this based on your model's structure
            list_of_names=['conv1', NodeType.PREFIX, 'conv2'],  # Ensure these layers exist in the model
            input_node=input_node,
            sep = ''
        )
        
        # Second branch (conv3 -> conv4), accessing layers correctly
        branch2_node = self.add_nodes_from_sequence(
            name_prefix='',  # Adjust this if your layers have a prefix (e.g., 'features.conv3')
            list_of_names=['conv3', NodeType.PREFIX, 'conv4'],  # Ensure these layers exist in the model
            input_node=input_node,
            sep = ''
        )
        
        # Concatenation operation after the two branches
        concat_node1 = self.create_node(node_type=NodeType.CONCAT)
        
        # Add directed edges from both branches to the concat node
        self.add_directed_edge(branch1_node, concat_node1)
        self.add_directed_edge(branch2_node, concat_node1)

        # Process after the first concat (conv5)
        after_concat1_node = self.add_nodes_from_sequence(
            name_prefix='',  # Adjust this if necessary
            list_of_names=['conv5'],  # Single layer conv5
            input_node=concat_node1,
            sep = ''
        )
        
        # Second concatenation: concatenate branch1 and after_concat1
        prefix_node2 = self.create_node(node_type=NodeType.PREFIX)
        
        concat_node2 = self.create_node(node_type=NodeType.CONCAT)
        self.add_directed_edge(branch1_node, prefix_node2)
        self.add_directed_edge(after_concat1_node, prefix_node2)
        self.add_directed_edge(prefix_node2, concat_node2)

        # Process after the second concat (conv6 -> pool -> fc)
        final_node = self.add_nodes_from_sequence(
            name_prefix='',  # Adjust this if necessary
            list_of_names=['conv6', NodeType.PREFIX, 'pool', 'fc'],  # Sequential layers conv6, pool, fc
            input_node=concat_node2,
            sep = ''
        )
        
        # Mark output
        self.add_directed_edge(final_node, self.create_node(node_type=NodeType.OUTPUT))
        
        return self


if __name__ == '__main__':
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from model_merger import ModelMerge
    from matching_functions import match_tensors_identity, match_tensors_zipit
    from copy import deepcopy
    model = SimpleConcatNet()  # Your custom model
    print(model)

    # Generate random data (batch_size=4, channels=3, 224x224)
    data_x = torch.rand(4, 3, 224, 224)
    data_y = torch.zeros(4)

    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=4)

    # Initialize the network
    model = SimpleConcatNet().eval()

    print(model)
    
    # Initialize the graph representation
    graph1 = SimpleConcatNetGraph(deepcopy(model)).graphify()
    graph2 = SimpleConcatNetGraph(deepcopy(model)).graphify()

    # Merging graphs (optional, just for demo if needed)

    # Draw the graph (last few nodes)
    # graph1.draw(nodes=range(len(graph1.G) - 10, len(graph1.G)))
    graph1.draw(
        save_path='./graphs/simple_concat_net_graph.png'
    )
    merge = ModelMerge(graph1, graph2)
    model3 = deepcopy(model)
    merge.transform(model3, dataloader, transform_fn=match_tensors_zipit)

    # Forward pass on the model
    print(model.eval().cuda()(data_x.cuda()))

    # Forward pass on the merged model (if merge is being used)
    print(merge(data_x.cuda()))
