import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pyinform
import json
import argparse

def main(nodes, time_steps, probability, metric, out_dir, inbox_cap, pull_top=5, seed=None):
	generator = np.random.default_rng(seed=seed)
	output = {}
	graph = initialize_ground_truth_graph(nodes, generator, seed)
	add_higher_order_relationships(graph, generator, num_relationships=3)
	save_graph(graph, out_dir, name="ground_truth_graph")
	save_higher_order_relationships(graph, out_dir)
	ground_truth_ranking = extract_rankings(graph, metric)
	save_rankings(ground_truth_ranking, metric, out_dir, name="ground_truth_ranking.txt")

	run_simuation(graph, time_steps, probability, generator, inbox_cap)
	series = extract_time_series(graph)
	save_time_series(series, out_dir)
	bin_series = binarize_time_series(series)
	save_time_series(bin_series, out_dir, name="binarized_synthetic_time_series.csv")
	
	te_graph = calculate_transfer_entropy(bin_series)
	save_graph(te_graph, out_dir,  name="te_graph")
	ranking = extract_rankings(te_graph, metric)
	save_rankings(ranking, metric, out_dir, name="te_ranking.txt")
	#trees = extract_trees(te_graph, ranking, pull_top)
	#save_trees(trees, ranking, metric, out_dir)
	rank_correlation = extract_rank_correlation(ground_truth_ranking, ranking)[0][1]
	output["rank correlation"] = rank_correlation
	print(rank_correlation)
	#graph_similarity = extract_graph_similarity(graph, te_graph)
	#output["graph similarity"] = graph_similarity
	#print(graph_similarity)
	save_output(output, out_dir)

def initialize_ground_truth_graph(nodes, generator, seed, source=0, path_weight=0.9):
	g = nx.scale_free_graph(nodes, alpha=0.05, gamma=0.41, seed=seed)
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
			graph[edge[0]][edge[1]]['weight'] = generator.random()
	for node in graph.nodes():
		graph.nodes[node]['Outbox'] = []
		graph.nodes[node]['Higher Order'] = []
	return graph

def add_higher_order_relationships(graph, generator, num_relationships=0, max_relationship_size=None, relationship_size_alpha=1.0, relationship_size_beta=1.0, max_exponent=1, exp_alpha=1.0, exp_beta=1.0):
	if num_relationships > 0:
		if max_relationship_size == None:
			max_relationship_size = graph.number_of_nodes()
		# generate relationship targets
		relationship_targets = generator.choice(graph.nodes(), size=num_relationships)
		# generate sizes
		relationship_sizes = (generator.beta(relationship_size_alpha, relationship_size_beta, size=num_relationships) * (max_relationship_size-3)).astype(np.int32) + 2
		# generate arrays
		for i in range(num_relationships):
			#initialize the array that describes this higher order relationship
			relationship = np.zeros(graph.number_of_nodes())
			# choose which nodes will participate in it, excluding the target node
			participants = generator.choice( [j for j in range(graph.number_of_nodes()) if j != relationship_targets[i]], size=relationship_sizes[i], replace=False)
			relationship[participants] = 1.0
			# add the higher order relationship to the releveant node
			graph.nodes()[relationship_targets[i]]['Higher Order'].append(relationship)

def add_inverse_weights(graph):
	for edge in graph.edges():
		graph[edge[0]][edge[1]]['inverse weight'] = 1.0 / graph[edge[0]][edge[1]]['weight']

def display_graph(graph):
	print(nx.adjacency_matrix(graph))
	for node in graph.nodes(data="Higher Order"):
		print(node)
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos)
	nx.draw_networkx_edges(graph, pos)
	nx.draw_networkx_labels(graph, pos)
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

def save_higher_order_relationships(graph, out_dir, name="higher_order_relationships"):
	with open(out_dir+"/"+name+".json", "w") as f:
		higher_order_relationship_dict = {t[0]: [list(a) for a in t[1]] for t in graph.nodes(data="Higher Order")}
		json.dump(higher_order_relationship_dict, f)

def load_graph(name="graph.txt"):
	# not tested
	return nx.read_weighted_edgelist(name)

def display_outboxes(graph):
	for node in graph.nodes():
		print(graph.nodes[node])

def run_simuation(graph, time_steps, probability, generator, inbox_cap, consider_higher_order = True):
	# time step 0; initialize time series with only original messages
	for node in graph.nodes():
		graph.nodes[node]['Outbox'].append(send_original_message(probability, generator))

	# time steps [1,time_steps]: extend time series with original and response messages
	for t in range(1,time_steps):
		for node in graph.nodes():
			graph.nodes[node]['Outbox'].append(send_original_message(probability, generator))
			for parent in graph.predecessors(node):
				incoming_message = graph.nodes[parent]['Outbox'][t-1]
				weight = graph[parent][node]['weight']
				graph.nodes[node]['Outbox'][t] += send_response_message(min(incoming_message, inbox_cap), weight, generator)
			graph.nodes[node]['Outbox'][t] += consider_higher_order * send_higher_order_response_message(graph, generator, node, t, probability, inbox_cap)

def send_original_message(probability, generator):
	return generator.binomial(1, probability)

def send_response_message(incoming_message, probability, generator):
	return generator.binomial(int(incoming_message), probability)

def send_higher_order_response_message(graph, generator, current_node, t, probability, inbox_cap):
	higher_order_contribution = 0
	for relationship in graph.nodes[current_node]['Higher Order']:
		contributors = np.nonzero(relationship)[0]
		contributions = [min(graph.nodes[i]['Outbox'][t-1], inbox_cap) for i in contributors if i != 0]
		geo_mean = np.prod(contributions)**(1/float(len(contributions)))
		higher_order_contribution += geo_mean
	return generator.binomial(int(higher_order_contribution), probability)

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

def extract_trees(te_graph, ranking, pull_top):
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

def extract_rank_correlation(true_rankings, test_rankings):
	true_rank_array = np.zeros(len(true_rankings))
	for centrality, node in true_rankings:
		true_rank_array[node] = centrality
	test_rank_array = np.zeros(len(test_rankings))
	for centrality, node in test_rankings:
		test_rank_array[node] = centrality
	rank_correlation = np.corrcoef(true_rank_array, test_rank_array)
	return rank_correlation

def extract_graph_similarity(true_graph, test_graph):
	return nx.graph_edit_distance(true_graph, test_graph)

def save_output(output, outdir, name="output"):
	with open(outdir+"/"+name+".json", "w") as f:
		json.dump(output, f)

if __name__ == '__main__':
	main(nodes=5, time_steps=20, probability=0.50, metric='betweenness', out_dir='output', pull_top=5, inbox_cap=12)