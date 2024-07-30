import torch
import torchvision.models as models
from copy import deepcopy
import sys
sys.path.append('./')
from graphs.base_graph import BIGGraph, NodeType


# class DenseNetGraph(BIGGraph):
#     def __init__(self, model, layer_name='features', head_name='classifier', num_layers=4):
#         super().__init__(model)
#         self.layer_name = layer_name
#         self.head_name = head_name
#         self.num_layers = num_layers


#     def add_dense_block_nodes(self, name_prefix, input_node):
#         dense_block = self.get_module(name_prefix)
#         num_layers = len([name for name in dense_block.named_children() if name[0].startswith('denselayer')])
#         previous_output_nodes = [input_node]

#         for i in range(1, 2 + 1):
#             layer_prefix = f'{name_prefix}.denselayer{i}'


#             shortcut_output_node = self.add_nodes_from_sequence(
#                 name_prefix='shortcut',
#                 list_of_names=[],
#                 input_node=previous_output_nodes[-1]
#             )


#             output_node = self.add_nodes_from_sequence(
#                 name_prefix=layer_prefix,
#                 list_of_names=['norm1', 'conv1', 'norm2', 'conv2', NodeType.CONCAT],
#                 input_node=previous_output_nodes[-1],
#             )
            

#             if len(previous_output_nodes)>1:
#                 for prev_node in previous_output_nodes:
#                     # concat_input_node = self.add_nodes_from_sequence(
#                     #     name_prefix=layer_prefix + 'concat',
#                     #     list_of_names=[NodeType.CONCAT],
#                     #     input_node=prev_node
#                     # )
#                     self.add_directed_edge(prev_node, output_node)

#             self.add_directed_edge(previous_output_nodes[-1], shortcut_output_node)
#             previous_output_nodes.append(output_node)

#             # self.add_directed_edge(previous_output_nodes[-1], output_node)


#         return previous_output_nodes[-1]

#     def add_transition_layer_nodes(self, name_prefix, input_node):
#         print(name_prefix)
#         transition_layers = self.get_module(name_prefix)

#         return self.add_nodes_from_sequence(name_prefix, ['norm', 'conv'], input_node)


#     # def add_layer_nodes(self, name_prefix, input_node):
#     #     source_node = input_node
#     #     dl = []
#     #     tl = []
#     #     for layer_index, block in enumerate(self.get_module(name_prefix)):
#     #         print(layer_index)
#     #         block_class = block.__class__.__name__
#     #         print('blockkkkkkkk',layer_index, block_class)
#     #         if block_class == '_DenseBlock':
#     #             dl.append('denseblock')
#     #             source_node = self.add_dense_block_nodes(name_prefix+ '.denseblock' + f'{len(dl)}', source_node)
#     #         elif block_class == '_Transition':
#     #             tl.append('transitionblock')
#     #             source_node = self.add_transition_layer_nodes(name_prefix+ '.transition' + f'{len(tl)}', source_node)
#     #         else:
#     #             print('hiiiiiiiiiii')
#     #             continue
        
#     #     return source_node

#     def graphify(self):
#         input_node = self.create_node(node_type=NodeType.INPUT)
#         input_node = self.add_nodes_from_sequence('', ['features.conv0', 'features.norm0'], input_node, sep='')

#         dl = []
#         tl = []
#         for layer_index, block in enumerate(self.get_module(self.layer_name)):
#             block_class = block.__class__.__name__
#             print('blockkkkkkkk',layer_index, block_class)
#             print(dl,tl)
#             if len(dl)==2:
#                 break
#             if block_class == '_DenseBlock':
#                 dl.append('denseblock')
#                 input_node = self.add_dense_block_nodes(self.layer_name + '.denseblock' + f'{len(dl)}', input_node)
#             elif block_class == '_Transition':
                
#                 tl.append('transitionblock')
#                 input_node = self.add_transition_layer_nodes(self.layer_name + '.transition' + f'{len(tl)}', input_node)
#             else:
#                 continue

#         input_node = self.add_nodes_from_sequence('', ['features.norm5', NodeType.PREFIX, self.head_name, NodeType.OUTPUT], input_node, sep='')
        
#         return self


        
        # for i in range(1, self.num_layers + 1):
        #     input_node = self.add_layer_nodes(f'{self.layer_name}', input_node)















        #     if i != self.num_layers:
        #         input_node = self.add_transition_layer_nodes(f'{self.layer_name}.transition{i}', input_node)
        # input_node = self.add_nodes_from_sequence('', ['features.norm5', NodeType.PREFIX, self.head_name, NodeType.OUTPUT], input_node, sep='')
        # input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX, 'avgpool', self.head_name, NodeType.OUTPUT], input_node, sep='')






























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
            
            output_node = self.add_nodes_from_sequence(
                name_prefix=layer_name,
                list_of_names=['norm1', 'conv1', 'norm2','conv2', NodeType.CONCAT],
                input_node=previous_output_nodes[-1],
            )
            if len(previous_output_nodes) > 1:
                for prev_node in previous_output_nodes:
                    self.add_directed_edge(prev_node, output_node)
            else:
                self.add_directed_edge(previous_output_nodes[-1], output_node)
            previous_output_nodes.append(output_node)
        return previous_output_nodes[-1]

    def add_transition_layer_nodes(self, name_prefix, input_node):
        transition_layer = self.get_module(name_prefix)
        # input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX], input_node)
        return self.add_nodes_from_sequence(name_prefix, ['norm', 'conv', NodeType.PREFIX], input_node)

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
                print('dense')
            elif block_class == '_Transition':
                input_node = self.add_transition_layer_nodes(f'{self.layer_name}.transition{tl}', input_node)
                tl+=1
                print('transition')
        input_node = self.add_nodes_from_sequence('', ['features.norm5', self.head_name, NodeType.OUTPUT], input_node, sep='')
        
        return self









from torchsummaryX import summary


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
    # unit test, nice
    # call from root directory with `python -m "graphs.resnet_graph"`
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.models.resnet as resnet
    import torchvision.models as models
    from model_merger import ModelMerge
    from matching_functions import match_tensors_identity, match_tensors_zipit
    from copy import deepcopy


    model3 = models.densenet121(pretrained=True).eval()
    graph1 = densenet121(deepcopy(model)).graphify()
    graph2 = densenet121(deepcopy(model)).graphify()
    # arch = summary(model3, torch.rand(1,3,224,224))
    graph1.draw(save_path='./densenet121_v41.png', nodes=range(400))
    merge = ModelMerge(graph1, graph2)
    merge.transform(model3, dataloader, transform_fn=match_tensors_zipit)

    # #graph1.draw(nodes=range(121))
    # graph1.draw(nodes=range(len(graph1.G)-20, len(graph1.G)))

    # print(model.cuda()(data_x.cuda()))
    # print(model3.cuda()(data_x.cuda()))
    # print(merge(data_x.cuda()))

