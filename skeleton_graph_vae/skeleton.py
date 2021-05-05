from kinematics import mjcf_parser
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

    
class Skeleton:
    def __init__(self, xml_file):
        self._shift = 5
        self._model = mjcf_parser.from_path(xml_file)
        self._parse()
        self._calculate_node_features()

    def _get_geom_size(self, g):
        if g.type == "sphere":
            l = g.size
        elif g.type == "capsule":
            l = np.sqrt(np.sum(np.square(g.fromto[:3] - g.fromto[3:])))
        return l

    def _get_node_features(self, node):
        if hasattr(node, 'name'):
            return node.name, node.type
        return node, "fake"

    def _is_end_body(self, body):
        return len(body.body) == 0

    def _parse(self):
        n_dummy = 1
        n_ee = 1
        root_body = self._model.worldbody.body[0]

        root_joint = root_body.joint[0]
        self.graph = nx.Graph()

        def _build_graph_recurse(parent_body, joint):
            # root_joint represent root pos and orientation
            nonlocal n_dummy
            nonlocal n_ee

            if self._is_end_body(parent_body):
                jc = f"ee_{n_ee}"
                n_ee += 1
                for g in parent_body.geom:
                    l = self._get_geom_size(g)
                    n1, t1 = self._get_node_features(joint)
                    n2, t2 = self._get_node_features(jc)
                    self.graph.add_edge(n1, n2, length=l)
                    self.graph.nodes[n1]["type"] = t1
                    self.graph.nodes[n2]["type"] = t2

            for b in parent_body.body:   
                if b.joint:
                    jc = b.joint[0]
                else:
                    jc = f"dummy_{n_dummy}"
                    n_dummy += 1
                for g in b.geom:
                    l = self._get_geom_size(g)
                    n1, t1 = self._get_node_features(joint)
                    n2, t2 = self._get_node_features(jc)
                    self.graph.add_edge(n1, n2, length=l)
                    self.graph.nodes[n1]["type"] = t1
                    self.graph.nodes[n2]["type"] = t2

                _build_graph_recurse(b, jc)
        
        _build_graph_recurse(root_body, root_joint)
       
    def _calculate_node_features(self):
        idx = 0
        for n, features in self.graph.nodes.data():
            if features["type"] == "free":
                self.graph.nodes[n]["pos"] = -1
                self.graph.nodes[n]["dof"] = 0
                self.graph.nodes[n]["mask"] = 0
                idx += 0
            elif features["type"] == "hinge":
                self.graph.nodes[n]["pos"] = idx
                self.graph.nodes[n]["dof"] = 1
                self.graph.nodes[n]["mask"] = 1

                idx += 1
            else:
                self.graph.nodes[n]["pos"] = -1
                self.graph.nodes[n]["dof"] = 0
                self.graph.nodes[n]["mask"] = 0

    def _to_edgelist(self):
        g = nx.to_directed(self.graph)
        g = nx.convert_node_labels_to_integers(g)
        self.edgeindex = [[x[0], x[1]] for x in g.edges]
        self.edge_feature = [g.edges[x[0], x[1]]["length"] for x in g.edges]
        self.pos = [g.nodes[n]["pos"] for n in g]
        self.dof = [g.nodes[n]["dof"] for n in g]

    def data_to_graph(self, raw_data, max_nodes):
        graph_data = dict()
        state = raw_data

        for n in self.graph.nodes:
            start = self._shift + self.graph.nodes.data()[n]['pos']
            end = start + self.graph.nodes.data()[n]['dof']
            if start < self._shift:
                x = np.zeros(1, dtype=np.float32)
            else:
                x = state[start:end]
            graph_data[n] = x

        x = torch.as_tensor([graph_data[n] for n in self.graph.nodes], dtype=torch.float32)
        node_feature = torch.zeros((max_nodes, 1), dtype=torch.float)
        node_feature[:x.shape[0]] = x

        g = nx.to_directed(self.graph)
        g = nx.convert_node_labels_to_integers(g)
        edge_index = torch.as_tensor([[x[0], x[1]] for x in g.edges], dtype=torch.long).t()
        edge_feature = torch.as_tensor([g.edges[x[0], x[1]]["length"] for x in g.edges], dtype=torch.float32)
        root_feature = torch.as_tensor(state[:5], dtype=torch.float)
        mask = torch.as_tensor([self.graph.nodes[n]["mask"] for n in self.graph.nodes], dtype=torch.float32)
        return Data(x=node_feature, edge_index=edge_index, 
                    edge_attr=edge_feature, root_feature=root_feature,
                    mask=mask)

    def attr_to_data(self, data, x):
        mask = np.array([self.graph.nodes[n]["mask"] for n in self.graph.nodes], 
                        dtype=np.float32)
        mask = np.expand_dims(mask, -1)
        x = x[torch.tensor(~(mask==0))]

        r = torch.tensor([0.5, 1, 0, 0, 0])
        r = data.root_feature
        x = torch.cat((r, x), dim=-1)
        x = x.detach().cpu().numpy()
        return x




        