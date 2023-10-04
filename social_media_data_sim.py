import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pyinform

def main(nodes, time_steps, probability, metric, in_dir, out_dir):
	graph = initialize_ground_truth_graph(nodes, in_dir)
	save_graph(graph, out_dir, name="ground_truth_graph")
	ground_truth_ranking = extract_rankings(graph, metric)
	save_rankings(ground_truth_ranking, metric, out_dir, name="ground_truth_ranking.txt")

	run_simuation(graph, time_steps, probability)
	series = extract_time_series(graph)
	save_time_series(series, out_dir)
	bin_series = binarize_time_series(series)
	save_time_series(bin_series, out_dir, name="binarized_synthetic_time_series.csv")
	
	te_graph = calculate_transfer_entropy(bin_series)
	save_graph(te_graph, out_dir,  name="te_graph")
	ranking = extract_rankings(te_graph, metric)
	save_rankings(ranking, metric, out_dir, name="te_ranking.txt")
	trees = extract_trees(te_graph, ranking)
	save_trees(trees, ranking, metric, out_dir)

def initialize_ground_truth_graph(nodes, in_dir, source=0, path_weight=0.9):
	g = nx.scale_free_graph(nodes, alpha=0.05, gamma=0.41)
	graph = nx.DiGraph(g)
	graph.remove_edges_from(nx.selfloop_edges(graph))
	paths = nx.single_source_shortest_path(graph, source)
	target = list(paths.keys())[-1]
	path = paths[list(paths.keys())[-1]]
	path_edges = [(path[i-1], path[i]) for i in range(1,len(path))]
	for edge in graph.edges():
		if edge in path_edges:
			graph[edge[0]][edge[1]]['weight'] = path_weight
		else:
			graph[edge[0]][edge[1]]['weight'] = np.random.random_sample()
	for node in graph.nodes():
		graph.nodes[node]['Outbox'] = []
	return graph

def add_inverse_weights(graph):
	for edge in graph.edges():
		graph[edge[0]][edge[1]]['inverse weight'] = 1.0 / graph[edge[0]][edge[1]]['weight']

def display_graph(graph):
	print(nx.adjacency_matrix(graph))
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos)
	nx.draw_networkx_edges(graph, pos)
	nx.draw_networkx_labels(graph, pos)
	#edge_labels = nx.get_edge_attributes(graph, "weight")
	#nx.draw_networkx_edge_labels(graph, pos, edge_labels)
	plt.show()
	plt.close()

def save_graph(graph, out_dir, name="graph"):
	nx.write_weighted_edgelist(graph, out_dir+'/'+name+'.txt')
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos)
	nx.draw_networkx_edges(graph, pos)
	nx.draw_networkx_labels(graph, pos)
	plt.savefig(out_dir+'/'+name+'.png')
	plt.close()

def load_graph(name="graph.txt"):
	# not tested
	return nx.read_weighted_edgelist(name)

def display_outboxes(graph):
	for node in graph.nodes():
		print(graph.nodes[node])

def run_simuation(graph, time_steps, probability):
	# time step 0; initialize time series with only original messages
	for node in graph.nodes():
		graph.nodes[node]['Outbox'].append(send_original_message(probability))

	# time steps [1,time_steps]: extend time series with original and response messages
	for t in range(1,time_steps):
		for node in graph.nodes():
			graph.nodes[node]['Outbox'].append(send_original_message(probability))
			for parent in graph.predecessors(node):
				incoming_message = graph.nodes[parent]['Outbox'][t-1]
				weight = graph[parent][node]['weight']
				graph.nodes[node]['Outbox'][t] += send_response_message(incoming_message, weight)

def send_original_message(probability):
	return np.random.choice([0.0, 1.0], p=[1.0-probability, probability])

def send_response_message(incoming_message, probability):
	return np.sum(np.random.choice([0.0, 1.0], size=int(incoming_message), p=[1.0-probability, probability]))

def extract_time_series(graph):
	return np.array([graph.nodes[node]['Outbox'] for node in graph.nodes])

def binarize_time_series(series, threshold = 0.0):
	return np.where(series > threshold, 1.0, 0.0)

def display_time_series(series):
	for s in series:
		print(s)

def save_time_series(series, out_dir, name = "synthetic_time_series.csv"):
	np.savetxt(out_dir+'/'+name, series, delimiter=',')

def calculate_transfer_entropy(series, history_length=1):
	l = list(product(range(len(series)),repeat=2))
	te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length) for source, target in l]
	te_graph = np.array(te_values_list).reshape((len(series),len(series)))
	threshold = np.mean(te_graph)
	te_graph = np.where(te_graph > threshold, te_graph, 0.0)
	te_graph = nx.DiGraph(te_graph)
	return te_graph

def extract_rankings(graph, metric = 'degree'):
	if metric == 'degree':
		ranking = nx.out_degree_centrality(graph)
	elif metric == 'betweenness':
		add_inverse_weights(graph)
		ranking = nx.betweenness_centrality(graph, weight = "inverse weight")
	elif metric == 'Katz':
		ranking = nx.katz_centrality(graph, weight = "weight")
	else:
		raise Exception("Invalid choice of metric. Valid choices are 'degree', 'betweenness', or 'Katz'")
	ranking = sorted([(ranking[key], key) for key in ranking.keys()], reverse = True)
	return ranking

def save_rankings(ranking, metric, out_dir, name="ranking.txt"):
	with open(out_dir+'/'+name, "w") as f:
		f.write(metric + ' centrality\n')
		for item in ranking:
			f.write("Node: {}, Centrality: {}\n".format(item[1], item[0]))

def extract_trees(te_graph, ranking, pull_top=5):
	trees = []
	for root in range(pull_top):
		trees.append(nx.bfs_tree(te_graph, ranking[root][1]))
		#trees.append(edge_bfs_tree(te_graph, ranking[root][1]))
	return trees

def edge_bfs_tree(graph, root):
	T = nx.DiGraph()
	T.add_node(root)
	edges_gen = nx.edge_bfs(graph, root)
	T.add_edges_from(edges_gen)
	return T

def save_trees(trees, ranking, metric, out_dir):
	for t in range(len(trees)):
		save_graph(trees[t], out_dir, name="tree_rank_{}_root_{}_{}_centrality_{}.txt".format(t+1, ranking[t][1], metric, ranking[t][0]))

if __name__ == '__main__':
	main(nodes=20, time_steps=20, probability=0.50, metric='betweenness', in_dir='input', out_dir='output')