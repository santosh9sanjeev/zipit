


import torch
import torchvision.models as models
from copy import deepcopy
import sys
sys.path.append('./')
from graphs.base_graph import BIGGraph, NodeType


class DenseNetGraph(BIGGraph):
    def __init__(self, model, layer_name='features', head_name='classifier'):
        super().__init__(model)
        self.layer_name = layer_name
        self.head_name = head_name


    def add_dense_block_nodes(self, name_prefix, input_node):
        dense_block = self.get_module(name_prefix)
        # input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX], input_node)
        previous_output_nodes = [input_node]
        for i, layer in enumerate(dense_block.children()):

            layer_name = f'{name_prefix}.denselayer{i+1}'
            if i==0:
                output_node = self.add_nodes_from_sequence(
                    name_prefix=layer_name,
                    list_of_names=['norm1', 'conv1','norm2', 'conv2'],
                    input_node=previous_output_nodes[-1],
                )

                output_node_1 = self.add_nodes_from_sequence(
                    name_prefix=layer_name,
                    list_of_names=[NodeType.CONCAT],
                    input_node=output_node,
                )
            else:
                output_node = self.add_nodes_from_sequence(
                    name_prefix=layer_name,
                    list_of_names=['norm1', 'conv1','norm2', 'conv2'],
                    input_node=output_node_1,
                )

                output_node_1 = self.add_nodes_from_sequence(
                    name_prefix=layer_name,
                    list_of_names=[NodeType.CONCAT],
                    input_node=output_node,
                )


            for prev_node in previous_output_nodes:
                self.add_directed_edge(prev_node, output_node_1)

            previous_output_nodes.append(output_node)
        previous_output_nodes.append(output_node_1)
        

        return previous_output_nodes[-1]

    def add_transition_layer_nodes(self, name_prefix, input_node):
        transition_layer = self.get_module(name_prefix)
        # input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX], input_node)
        return self.add_nodes_from_sequence(name_prefix, ['norm', NodeType.PREFIX, 'conv'], input_node)

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        input_node = self.add_nodes_from_sequence('', ['features.conv0', 'features.norm0'], input_node, sep='')

        # Iterate through the layers, starting from DenseBlock
        dl = 1
        tl = 1
        for i, block in enumerate(self.get_module(self.layer_name)):
            block_class = block.__class__.__name__
            if block_class == '_DenseBlock':
                input_node = self.add_dense_block_nodes(f'{self.layer_name}.denseblock{dl}', input_node)
                dl+=1
            elif block_class == '_Transition':
                input_node = self.add_transition_layer_nodes(f'{self.layer_name}.transition{tl}', input_node)
                tl+=1
        input_node = self.add_nodes_from_sequence('', ['features.norm5', NodeType.PREFIX, self.head_name, NodeType.OUTPUT], input_node, sep='')
        
        return self









from torchsummaryX import summary
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models.resnet as resnet
import torchvision.models as models
from model_merger import ModelMerge
from matching_functions import match_tensors_identity, match_tensors_zipit
from copy import deepcopy

def densenet121(model):
    return DenseNetGraph(model)


if __name__ == '__main__':
    # Example usage
    from torch.utils.data import TensorDataset, DataLoader

    model = models.densenet121(pretrained=False).eval()
    state_dict = model.state_dict()
    data_x = torch.rand(4, 3, 224, 224)
    data_y = torch.zeros(4)
    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=4)



    model3 = models.densenet121(pretrained=True).eval()
    graph1 = densenet121(deepcopy(model)).graphify()
    graph2 = densenet121(deepcopy(model)).graphify()
    arch = summary(model3, torch.rand(1,3,224,224))
    # print(arch)
    graph1.draw(save_path='./densenet121_v6.png', nodes=range(400))
    merge = ModelMerge(graph1, graph2)
    merge.transform(model3, dataloader, transform_fn=match_tensors_identity)

    # #graph1.draw(nodes=range(121))
    # graph1.draw(nodes=range(len(graph1.G)-20, len(graph1.G)))

    print(model.cuda()(data_x.cuda()))
    print(model3.cuda()(data_x.cuda()))
    print(merge(data_x.cuda()))