import torch
# from torch_geometric.nn.conv


class Graph():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.h = torch.clone(x)
        self.h_outputs = []
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.original_edge_attr = torch.clone(edge_attr)
        self.h_edges = []


class SceneGraph():
    def __init__(self, x, edge_index, edge_attr, type, edge_weight,remap):
        self.x = x
        self.h = torch.clone(x)
        self.h_outputs = []

        self.x_edge = edge_attr
        self.h_edge = torch.clone(edge_attr)
        self.h_edge_outputs = []

        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.type = type
        self.remap = remap
    def reset_graph(self):
        self.h_outputs = []
        self.h_edge_outputs = []


class SceneGraph_noweight():
    def __init__(self, x, edge_index, edge_attr, type,remap):
        self.x = x
        self.h = torch.clone(x)
        self.h_outputs = []

        self.x_edge = edge_attr
        self.h_edge = torch.clone(edge_attr)
        self.h_edge_outputs = []

        self.edge_index = edge_index

        self.type = type
        self.remap = remap
    def reset_graph(self):
        self.h_outputs = []
        self.h_edge_outputs = []

class Knowledge_Matrix():
    def __init__(self, node_knowledge, edge_knowledge):
        self.node_knowledge = node_knowledge
        self.edge_knowledge = edge_knowledge

    def update_knowledge(self, node_knowledge, edge_knowledge):
        self.node_knowledge = node_knowledge
        self.edge_knowledge = edge_knowledge
