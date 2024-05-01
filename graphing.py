import networkx as nx
import matplotlib.pyplot as plt

# Construct training graph
with open('train.txt', 'r') as file:
    G_train = nx.Graph()
    lines = file.readlines()
    for line in lines:
        # Split the line by commas
        nodes = [node.strip() for node in line.strip().split(',')]
        G_train.add_nodes_from(nodes)
        G_train.add_edge(*nodes)

# Construct Testing graph
with open('test.txt', 'r') as file:
    G_test = nx.Graph()
    lines = file.readlines()
    for line in lines:
        # Split the line by commas
        nodes = [node.strip() for node in line.strip().split(',')]
        G_test.add_nodes_from(nodes)
        G_test.add_edge(*nodes)

# Construct Universe graph
with open('full.txt', 'r') as file:
    G_uni = nx.Graph()
    lines = file.readlines()
    for line in lines:
        # Split the line by commas
        nodes = [node.strip() for node in line.strip().split(',')]
        G_uni.add_nodes_from(nodes)
        G_uni.add_edge(*nodes)

print("Number of Training Nodes:", G_train.order())
print("Number of Training Edges:", G_train.size())
print("Average Degree of Training data:", float(G_train.size())/ G_train.order())

print("Number of Testing Nodes:", G_test.order())
print("Number of Testing Edges:", G_test.size())
print("Average Degree of Testing data:", float(G_test.size())/ G_test.order())

print("Number of Nodes:", G_uni.order())
print("Number of Edges:", G_uni.size())
print("Average Degree:", float(G_uni.size())/ G_uni.order())

