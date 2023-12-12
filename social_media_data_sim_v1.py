import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pyinform
import json
import argparse
import os
import pandas as pd
import time
from scipy.stats import ttest_ind

def main():
	parser = initialize_parser()
	args = parser.parse_args()
	start_time = time.time()
	_, _, _ = experiment(args)
	print("Elapsed time = {} seconds".format(time.time() - start_time))

def initialize_parser():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--seed', type=int)
	parser.add_argument('--trials', type=int)
	parser.add_argument('--nodes', type=int, default=5)
	parser.add_argument('--min_nodes', type=int, default=3)
	parser.add_argument('--edges', type=int)
	parser.add_argument('--source', type=int, default=0)
	parser.add_argument('--time_steps', type=int, default=20)
	parser.add_argument('--inbox_cap', type=int, default=12)
	parser.add_argument('--threshold', type=float, default=0.0)
	parser.add_argument('--history_length', type=int, default=1)
	parser.add_argument('--metric', default="Katz")
	parser.add_argument('--pull_top', type=int, default=5)
	parser.add_argument('--out_dir', default="output")
	parser.add_argument('--save_output', type=int, default=1)
	parser.add_argument('--timeout', type=int, default=1)
	parser.add_argument('--scale_free_alpha', type=float, default=0.05)
	parser.add_argument('--scale_free_beta', type=float, default=0.54)
	parser.add_argument('--scale_free_gamma', type=float, default=0.41)
	parser.add_argument('--path_weight', type=float, default=0.9)
	parser.add_argument('--num_relationships', type=int, default=3)
	parser.add_argument('--min_relationships', type=int, default=1)
	parser.add_argument('--max_relationship_size', type=int)
	parser.add_argument('--relationship_size_alpha', type=float, default=1.0)
	parser.add_argument('--relationship_size_beta', type=float, default=1.0)
	parser.add_argument('--probability', type=float, default=0.5)
	parser.add_argument('--higher_order_probability', type=float, default=0.5)
	parser.add_argument('--consider_higher_order', type=int, default=1)
	return parser

def initialize_ground_truth_graph(nodes, edges, generator, seed, alpha, beta, gamma, source, path_weight):
	# initialize a scale free graph of the desired size to map out the social interactions
	graph = watts_strogatz_graph_wrapper(nodes, edges, 0.5, seed, generator)
	paths = nx.single_source_shortest_path(graph, source)
	target = list(paths.keys())[-1]
	path = paths[list(paths.keys())[-1]]
	# set the weights of edges along the path to a high value to create a cascade of influence
	path_edges = [(path[i-1], path[i]) for i in range(1, len(path))]
	for edge in graph.edges():
		if edge in path_edges:
			graph[edge[0]][edge[1]]['weight'] = path_weight
		else:
			graph[edge[0]][edge[1]]['weight'] = generator.random()*path_weight
	# add node data to track social activity and higher order relationships
	for node in graph.nodes():
		graph.nodes[node]['Outbox'] = []
		graph.nodes[node]['Higher Order'] = []
	return graph

def watts_strogatz_graph_wrapper(nodes, edges, probability, seed, generator):
	# handle the boundary cases of tree graphs or dense graphs
	if edges == nodes-1:
		t = nx.from_prufer_sequence([generator.choice(range(nodes)) for i in range(nodes-2)])
		return nx.DiGraph(t)
	if edges == nodes**2:
		g = nx.complete_graph(nodes)
		return nx.DiGraph(g)
	# obtain number of nearest neighbor connections from desired graph density
	k = int(edges/nodes)+1
	G = nx.watts_strogatz_graph(nodes, k, probability, seed)
	# guarantee connectivity of G and convert it to a DiGraph
	augmentation = list(nx.k_edge_augmentation(G, k=1))
	G.add_edges_from(augmentation)
	G = nx.DiGraph(G)
	return G

def add_higher_order_relationships(graph, generator, num_relationships, max_relationship_size, relationship_size_alpha, relationship_size_beta):
	if num_relationships > 0:
		if max_relationship_size == None:
			max_relationship_size = graph.number_of_nodes()
		else:
			max_relationship_size = min(graph.number_of_nodes(), max_relationship_size)
		# generate relationship targets
		relationship_targets = generator.choice(graph.nodes(), size=num_relationships)
		# generate relationship sizes
		relationship_sizes = (generator.beta(relationship_size_alpha, relationship_size_beta, size=num_relationships) * (max_relationship_size-3)).astype(np.int32) + 2
		# generate arrays representing relationships
		for i in range(num_relationships):
			relationship = np.zeros(graph.number_of_nodes())
			# choose which nodes will participate in it, excluding the target node
			participants = generator.choice( [j for j in range(graph.number_of_nodes()) if j != relationship_targets[i]], size=relationship_sizes[i], replace=False)
			relationship[participants] = 1.0
			# add the higher order relationship to the data of the target node
			graph.nodes()[relationship_targets[i]]['Higher Order'].append(relationship)

def add_inverse_weights(graph):
	# add a reciprocal weight to each edge, for use with betweeness centrality
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
	'''
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos)
	nx.draw_networkx_edges(graph, pos)
	nx.draw_networkx_labels(graph, pos)
	plt.savefig(out_dir+'/'+name+'.png')
	plt.close()
	'''

def save_higher_order_relationships(graph, out_dir, name="higher_order_relationships"):
	with open(out_dir+"/"+name+".json", "w") as f:
		higher_order_relationship_dict = {t[0]: [list(a) for a in t[1]] for t in graph.nodes(data="Higher Order")}
		json.dump(higher_order_relationship_dict, f)

def display_outboxes(graph):
	for node in graph.nodes():
		print(graph.nodes[node])

def run_simuation(graph, time_steps, probability, higher_order_probability, generator, inbox_cap, consider_higher_order):
	# time step 0; initialize time series with only original messages
	for node in graph.nodes():
		graph.nodes[node]['Outbox'].append(send_original_message(probability, generator))

	# time steps 1 through time_steps: extend time series with original and response messages
	for t in range(1,time_steps):
		for node in graph.nodes():
			# original messages
			graph.nodes[node]['Outbox'].append(send_original_message(probability, generator))
			# messages responding to messages sent by parent nodes in the previous time step
			for parent in graph.predecessors(node):
				incoming_message = graph.nodes[parent]['Outbox'][t-1]
				weight = graph[parent][node]['weight']
				graph.nodes[node]['Outbox'][t] += send_response_message(min(incoming_message, inbox_cap), weight, generator)
			# messages sent as a result of the influence of higher order relationships
			graph.nodes[node]['Outbox'][t] += consider_higher_order * send_higher_order_response_message(graph, generator, node, t, probability, inbox_cap)

def send_original_message(probability, generator):
	# a 1 corresponds to sending a message of interest, a 0 corresponds to not sending a message of interest
	return generator.binomial(1, probability)

def send_response_message(incoming_message, probability, generator):
	# a 1 corresponds to sending a message of interest, a 0 corresponds to not sending a message of interest
	return generator.binomial(int(incoming_message), probability)

def send_higher_order_response_message(graph, generator, current_node, t, probability, inbox_cap):
	higher_order_contribution = 0
	# for each higher order relationship, take the geometric mean of the contributing nodes's messages in the previous time step
	for relationship in graph.nodes[current_node]['Higher Order']:
		contributors = np.nonzero(relationship)[0]
		contributions = [min(graph.nodes[i]['Outbox'][t-1], inbox_cap) for i in contributors if i != 0]
		geo_mean = np.prod(contributions)**(1/float(len(contributions)))
		higher_order_contribution += geo_mean
	# a 1 corresponds to sending a message of interest, a 0 corresponds to not sending a message of interest
	return generator.binomial(int(higher_order_contribution), probability)

def extract_time_series(graph):
	return np.array([graph.nodes[node]['Outbox'] for node in graph.nodes])

def binarize_time_series(series, threshold):
	# binarize the time series based on a threshold of messages sent, for use with transfer entropy
	return np.where(series > threshold, 1.0, 0.0)

def display_time_series(series):
	for s in series:
		print(s)

def save_time_series(series, out_dir, name = "synthetic_time_series.csv"):
	np.savetxt(out_dir+'/'+name, series, delimiter=',')

def calculate_transfer_entropy(series, history_length):
	# calculate the transfer entropy between each pair of time series
	l = list(product(range(len(series)),repeat=2))
	#te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length) for source, target in l]
	te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length, condition=[series[i] for i in range(len(series)) if (i!=source and i!=target)]) for source, target in l]
	# use the transfer entropy values as edge weights to construct an adjacency matrix for a directed graph
	te_graph = np.array(te_values_list).reshape((len(series),len(series)))
	#threshold = np.mean(te_graph)
	#te_graph = np.where(te_graph >= threshold, te_graph, 0.0)
	te_graph = nx.DiGraph(te_graph)
	return te_graph

def extract_rankings(graph, metric):
	# calculate the specified centrality measure for each node
	if metric == 'degree':
		ranking = nx.out_degree_centrality(graph)
	elif metric == 'betweenness':
		add_inverse_weights(graph)
		ranking = nx.betweenness_centrality(graph, weight = "inverse weight")
		#ranking = nx.betweenness_centrality(graph)
	elif metric == 'Katz':
		g = graph.reverse()
		try:
			ranking = nx.katz_centrality(g, weight = "weight")
		except:
			ranking = dict(zip(g.nodes, np.zeros(g.number_of_nodes())))
		#ranking = nx.katz_centrality(graph, weight = "weight")
	elif metric == 'closeness':
		add_inverse_weights(graph)
		ranking = nx.closeness_centrality(graph, distance = "inverse weight")
		#ranking = nx.closeness_centrality(graph)
	elif metric == 'eigenvector':
		ranking = nx.eigenvector_centrality(graph, max_iter=100000, weight = "weight")
		#ranking = nx.eigenvector_centrality(graph, max_iter=1000)
	else:
		raise Exception("Invalid choice of metric. Valid choices are 'degree', 'betweenness', or 'Katz'")
	# sort the nodes by their centrality score
	ranking = sorted([(ranking[key], key) for key in ranking.keys()], reverse = True)
	return ranking

def save_rankings(ranking, metric, out_dir, name="ranking.txt"):
	with open(out_dir+'/'+name, "w") as f:
		f.write(metric + ' centrality\n')
		for item in ranking:
			f.write("Node: {}, Centrality: {}\n".format(item[1], item[0]))

def extract_trees(te_graph, ranking, pull_top):
	trees = []
	# for each of the top ranked nodes, calculate the breadth first search tree
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
	# calculate the rank correlation between the centrality scores of the ground truth graph and the te graph 
	true_rank_array = np.zeros(len(true_rankings))
	for centrality, node in true_rankings:
		true_rank_array[node] = centrality
	test_rank_array = np.zeros(len(test_rankings))
	for centrality, node in test_rankings:
		test_rank_array[node] = centrality
	rank_correlation = np.corrcoef(true_rank_array, test_rank_array)
	return rank_correlation

def extract_graph_similarity(true_graph, test_graph, timeout):
	gen = nx.optimize_graph_edit_distance(true_graph, test_graph)
	t = 0
	for s in gen:
		if t > timeout:
			break
		min_s = s
		t += 1
	return min_s

def experiment(args):
	# initialize data frames
	test_rc_results = pd.DataFrame()
	control_rc_results = pd.DataFrame()
	
	# loop thorugh all trials for each graph size
	for nodes in range(3, args.nodes+1):
		print("Nodes = {}".format(nodes))
		t, f = trials(args, nodes, nodes*2, args.num_relationships, args.time_steps, args.probability, args.path_weight, args.higher_order_probability, args.inbox_cap)
		test_rc_results[nodes] = t
		control_rc_results[nodes] = f

	# save results
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)
	test_rc_results.to_json(path_or_buf=args.out_dir+"/test_rc.json")
	control_rc_results.to_json(path_or_buf=args.out_dir+"/control_rc.json")

	test_means = test_rc_results.mean()
	test_std_devs = test_rc_results.std()
	control_means = control_rc_results.mean()
	control_std_devs = control_rc_results.std()
	p_values = {}
	df = {}
	for col in test_rc_results:
		p_values[col] = ttest_ind(test_rc_results[col], control_rc_results[col], equal_var=False).pvalue
		df[col] = ttest_ind(test_rc_results[col], control_rc_results[col], equal_var=False).df
	results = pd.DataFrame({"test means": test_means, "test std devs": test_std_devs, "control means": control_means, "control std devs": control_std_devs, "p values": p_values, "dfs": df})
	results.to_csv(path_or_buf=args.out_dir+"/results.csv")
	print(results)
	
	return test_rc_results, control_rc_results, results

def trials(args, nodes, edges, relationships, time_steps, probability, path_weight, higher_order_probability, inbox_cap):
	t_rank_correlation = np.zeros(args.trials)
	f_rank_correlation = np.zeros(args.trials)
	t = 0
	while t < args.trials:
		# print progress for user, define a dict to store output, and initialize the random number generator
		#print("Number of Nodes = {}, Number of Edges = {}, Number of Higher Order Relationships = {}, Trial {}".format(nodes, edges, relationships, t))
		generator = np.random.default_rng(seed=args.seed)
			
		# define the multi-agent system for this trial
		graph = initialize_ground_truth_graph(nodes, edges, generator, args.seed, args.scale_free_alpha, args.scale_free_beta, args.scale_free_gamma, args.source, path_weight)
		add_higher_order_relationships(graph, generator, relationships, args.max_relationship_size, args.relationship_size_alpha, args.relationship_size_beta)
		ground_truth_ranking = extract_rankings(graph, args.metric)

		# run the simulatione once with higher order relationships
		run_simuation(graph, time_steps, probability, higher_order_probability, generator, inbox_cap, True)
		t_series = extract_time_series(graph)
		t_bin_series = binarize_time_series(t_series, args.threshold)

		# construct transfer entropy netowrk and extract rankings from the simulation with higher order relationships
		t_te_graph = calculate_transfer_entropy(t_bin_series, args.history_length)
		if t_te_graph.size() == 0:
			continue
		t_ranking = extract_rankings(t_te_graph, args.metric)

		# collect stats for the transfer entropy network with higher order relationships
		t_rank_correlation[t] = extract_rank_correlation(ground_truth_ranking, t_ranking)[0][1]
			
		# empty the graph's message history and run the simulation once without higher order relationships
		for node in graph.nodes():
			graph.nodes[node]['Outbox'] = []
		run_simuation(graph, time_steps, probability, higher_order_probability, generator, inbox_cap, False)
		f_series = extract_time_series(graph)
		f_bin_series = binarize_time_series(f_series, args.threshold)
						
		# construct transfer entropy netowrk and extract rankings from the simulation without higher order relationships
		f_te_graph = calculate_transfer_entropy(f_bin_series, args.history_length)
		if f_te_graph.size() == 0:
			continue
		f_ranking = extract_rankings(f_te_graph, args.metric)

		# collect stats for the transfer entropy network without higher order relationships
		f_rank_correlation[t] = extract_rank_correlation(ground_truth_ranking, f_ranking)[0][1]

		# if output files for each trial are desired, establish a distinct output directory for each trial
		if(args.save_output):
			trial_out_dir = args.out_dir + "_nodes_"+str(nodes) + "_edges_"+str(edges) + "_relationships_"+str(relationships) + "/trial_"+str(t)
			if not os.path.exists(trial_out_dir):
				os.makedirs(trial_out_dir)

			save_graph(graph, trial_out_dir, name="ground_truth_graph")
			save_higher_order_relationships(graph, trial_out_dir)
			save_rankings(ground_truth_ranking, args.metric, trial_out_dir, name="ground_truth_rankings.txt")
			
			save_time_series(t_series, trial_out_dir, name = "synthetic_time_series_higher_order.csv")
			save_time_series(t_bin_series, trial_out_dir, name = "binarized_synthetic_time_series_higher_order.csv")
			
			save_graph(t_te_graph, trial_out_dir,  name="te_graph_higher_order")
			save_rankings(t_ranking, args.metric, trial_out_dir, name="te_ranking_higher_order.txt")
			
			save_time_series(f_series, trial_out_dir, name = "synthetic_time_series_control.csv")
			save_time_series(f_bin_series, trial_out_dir, name = "binarized_synthetic_time_series_control.csv")
			
			save_graph(f_te_graph, trial_out_dir,  name="te_graph_control")
			save_rankings(f_ranking, args.metric, trial_out_dir, name="te_ranking_control.txt")
			
			output = {}
			output["higher order rank correlation"] = t_rank_correlation
			output["control rank correlation"] = f_rank_correlation
			output["args"] = vars(args)
			save_output(output, trial_out_dir)

		t += 1

	return (t_rank_correlation, f_rank_correlation)

def save_output(output, outdir, name="output"):
	with open(outdir+"/"+name+".json", "w") as f:
		json.dump(output, f)

if __name__ == '__main__':
	main()