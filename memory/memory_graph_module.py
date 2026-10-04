import networkx as nx
import matplotlib.pyplot as plt

class MemoryGraph:
    def __init__(self):
        self.G = nx.Graph()

    def add_node(self, node):
        self.G.add_node(node)

    def add_edge(self, node1, node2):
        self.G.add_edge(node1, node2)

    def draw_graph(self):
        nx.draw(self.G, with_labels=True)
        plt.show()
