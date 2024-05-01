import math
import linkpred
import itertools
import numpy as np
import networkx as nx
from sklearn import metrics
from graphing import G_train, G_test


# Algorithm for LLP
def llp(graph):

    # Equation 10
    n = {}
    for i in graph.nodes():
        for j in graph.nodes():
            first_order = list(nx.all_neighbors(graph, i))
            second_order = list(nx.node_boundary(graph, first_order + [i])) + first_order
            if i == j:
                n[i] = graph.degree[i]
            elif j in first_order:
                n[i] = len(list(nx.common_neighbors(graph, i, j))) + 1
            elif j in second_order:
                n[i] = len(list(nx.common_neighbors(graph, i, j)))
            else:
                n[i] = 0

    # Equation 11
    un_ij = {}
    for non_edge in nx.non_edges(graph):
        n_iz = n[non_edge[0]]
        n_jz = n[non_edge[1]]
        n_i_sum = 0
        n_j_sum = 0
        if n_iz > 0:
            un_ij[non_edge] = n_iz
            n_i_sum += n_iz
        elif n_jz > 0:
            un_ij[non_edge] = n_jz
            n_j_sum += n_jz

    # Equation 13
    n_i_bar = n_i_sum/len(un_ij)
    n_j_bar = n_j_sum/len(un_ij)

    # Equation 12
    num_ij_prod = 0
    denom_i_sum = 0
    denom_j_sum = 0

    corr = {}
    for non_edge in nx.non_edges(graph):
        n_i_diff = n[non_edge[0]] - n_i_bar
        n_j_diff = n[non_edge[1]] - n_j_bar

        num_ij_prod += n_i_diff * n_j_diff

        denom_i_sum += n_i_diff ** 2
        denom_j_sum += n_j_diff ** 2

        corr[non_edge] = num_ij_prod/(0.00001 + math.sqrt(denom_i_sum) * math.sqrt(denom_j_sum))

    # Equation 14
    dicn = {}
    for non_edge in nx.non_edges(graph):
        dicn[non_edge] = (1 + len(list(nx.common_neighbors(graph, non_edge[0], non_edge[1])))) * (1 + corr[non_edge])

    return dicn

# Calculate the DICN for training data
llp_scores = llp(G_train)

# Generate ground testing data
ground_truth = {}
for edge in (G_test.edges() or nx.non_edges(G_test)):
    if edge in G_train.edges():
        ground_truth[edge] = 1
    else:
        ground_truth[edge] = 0

# Extract ground truth labels
labels = list(ground_truth.values())

# Extract predicted values
prob = []
for i in ground_truth.keys():
    if i in llp_scores.keys():
        prob.append(llp_scores[i] * -1)
    else:
        prob.append(0)

# Compute ROC AUC for LLP
llp_auc = metrics.roc_auc_score(np.array(labels), np.array(prob))
print("LLP AUC Score:", llp_auc)

########################################################################################################################

# Create universe and test sets for LinkPred Katz algorithm
nodes = list(G_train.nodes())
nodes.extend(list(G_test.nodes()))
test_set = [linkpred.evaluation.Pair(i) for i in G_test.edges()]
universe_set = set([linkpred.evaluation.Pair(i) for i in itertools.product(nodes, nodes) if i[0] != i[1]])
G_universe = nx.compose(G_train, G_test)

# Compute ROC AUC for Katz Algorithm using LinkPred library
k_scores = (linkpred.predictors.Katz(G_train, excluded=None)).predict()
k_eval = linkpred.evaluation.EvaluationSheet(k_scores, test_set, universe_set)
print("Katz AUC Score:", metrics.auc(k_eval.fallout(), k_eval.recall()))
