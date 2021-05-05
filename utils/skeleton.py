from kinematics import mjcf_parser
import networkx as nx
import numpy as np

class Skeleton:
    def __init__(self, xml_file):
        self._model = mjcf_parser.from_path(xml_file)
        self._parse()
        self._to_edgelist()

    def _get_geom_size(self, g):
        if g.type == "sphere":
            l = g.size
        elif g.type == "capsule":
            l = np.sqrt(np.sum(np.square(g.fromto[:3] - g.fromto[3:])))
        return l

    def _get_node_name(self, node):
        if hasattr(node, 'name'):
            return node.name
        return node

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
                n1 = self._get_node_name(joint)
                n2 = jc
                for g in parent_body.geom:
                    l = self._get_geom_size(g)
                    n1 = self._get_node_name(joint)
                    n2 = self._get_node_name(jc)
                    self.graph.add_edge(n1, n2, length=l)

            for b in parent_body.body:   
                if b.joint:
                    jc = b.joint[0]
                else:
                    jc = f"dummy_{n_dummy}"
                    n_dummy += 1
                for g in b.geom:
                    l = self._get_geom_size(g)
                    n1 = self._get_node_name(joint)
                    n2 = self._get_node_name(jc)
                    self.graph.add_edge(n1, n2, length=l)
                _build_graph_recurse(b, jc)
        
        _build_graph_recurse(root_body, root_joint)
        for n in self.graph.nodes:
            if "dummy" in n or "ee" in n:
                self.graph.nodes[n]["m"] = 0.
            else:
                self.graph.nodes[n]["m"] = 1.

    def _to_edgelist(self):
        g = nx.to_directed(self.graph)
        g = nx.convert_node_labels_to_integers(g)
        self.edgeindex = [[x[0], x[1]] for x in g.edges]
        self.edge_feature = [g.edges[x[0], x[1]]["length"] for x in g.edges]
        self.node_feature = [g.nodes[n]["m"] for n in g]


        