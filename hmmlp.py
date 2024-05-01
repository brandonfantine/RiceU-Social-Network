# Algorithm for HMMLP
def hmmlp(graph, a, b):
    r = {}
    for i in graph.nodes():
        for j in graph.nodes():
            if i != j:
                k_i = graph.degree[i]
                k_j = graph.degree[j]

                # Using assumption that the probability of connection is related to degree of nodes
                # Using assumption that no node in the graph can have a r_ij of 1
                rHat = (k_i * k_j)/graph.size()

                # if i and j in rice:
                #     rHat += rice.size()/graph.size()

                # Estimate distance using rHat
                distance = (b * k_i * k_j) * ((rHat ** (-1/a)) - 1)

                # Calculate r for all combinations of points using the estimated dsitance
                r[(i, j)] = ((1 + distance)/(b * k_i * k_j)) ** (a * -1)
    return r
# Set alpha > 1
a = 2

# beta = (1/k_max)^2
k_max = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]
b = (1/k_max) ** 2

# Compute probabilities using the HMMLP on the training data
r_scores = hmmlp(G, a, b)
