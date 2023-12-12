import argparse
import numpy as np
import time
from scipy.sparse.csgraph import shortest_path
from itertools import product
import pyinform
import json
import os
import pandas as pd
from scipy.stats import ttest_ind
from numba import njit
#import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx

def main():
	parser = initialize_parser()
	args = parser.parse_args()
	start_time = time.time()
	#experiment(
	#	args.out_dir, args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
	#	args.max_path_weight, args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity, args.inbox_cap)
	ex_rewire(
		args.out_dir, args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
		args.max_path_weight, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity, args.inbox_cap)
	ex_max_weight(
		args.out_dir, args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
		args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity, args.inbox_cap)
	ex_emission(
		args.out_dir, args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
		args.max_path_weight, args.rewire_probability, args.max_relationship_size, args.time_steps, args.higher_order_sensitivity, args.inbox_cap)
	ex_hos(
		args.out_dir, args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
		args.max_path_weight, args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, args.inbox_cap)
	ex_inbox(
		args.out_dir, args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
		args.max_path_weight, args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity)
	print("Elapsed time = {} seconds".format(time.time() - start_time))
	#sample_graphs(5, 10, args.max_path_weight, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity, args.inbox_cap, args.out_dir)

def initialize_parser():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	
	parser.add_argument('--out_dir', default="output/temp_test")
	parser.add_argument('--min_relationships', type=int, default=1)
	parser.add_argument('--max_relationships', type=int, default=10)
	parser.add_argument('--max_relationship_size', type=int, default=5)
	parser.add_argument('--min_nodes', type=int, default=5)
	parser.add_argument('--max_nodes', type=int, default=50)
	parser.add_argument('--trials', type=int, default=50)

	parser.add_argument('--time_steps', type=int, default=200) # 200
	parser.add_argument('--inbox_cap', type=int, default=1) # 1
	parser.add_argument('--emission_probability', type=float, default=0.1) # 0.1
	parser.add_argument('--higher_order_sensitivity', type=float, default=0.1) # 0.1
	parser.add_argument('--max_path_weight', type=float, default=0.5) # 0.5
	parser.add_argument('--rewire_probability', type=float, default=0.3) #0.3

	return parser

def Generate_Watts_Strogatz_Graph(generator, nodes, path_weight, rewire_probability):
	g, w = initialize_ring_lattice(generator, nodes, path_weight)
	g = rewire_edges(generator, g, rewire_probability)
	g = restore_connectivity(g)
	return g*w

#@njit
def initialize_ring_lattice(generator, nodes, path_weight):
	# initialize empty graph, and weights for later use
	g = np.zeros((nodes, nodes))
	w = generator.uniform(low=0.01, high=path_weight, size=(nodes, nodes))
	# set the graph to a ring lattice with k=1
	for n in range(nodes):
		g[n][(n+1)%nodes] = 1.0
		g[n][n-1] = 1.0
	return g, w

#@njit
def rewire_edges(generator, g, rewire_probability):
	# extract edges and temporarily set diagonal entries to 1
	e_r, e_c = g.nonzero()
	e = zip(e_r, e_c)
	np.fill_diagonal(g, 1)

	# rewire edges and restore diagonal entries to 0
	for edge in e:
		c = generator.binomial(1, rewire_probability)
		if(c):
			g[edge[0]][edge[1]] = 0.0
			g[edge[1]][edge[0]] = 0.0
			target = generator.permutation((g[edge[0]]==0).nonzero()[0])[0]
			g[edge[0]][target] = 1.0
			g[target][edge[0]] = 1.0
	np.fill_diagonal(g, 0)
	return g

def restore_connectivity(g):
	# check for disconnected nodes and reconnect them
	l = shortest_path(g, indices=0)
	while(np.any(l==np.inf)):
		t = (l==np.inf).nonzero()[0][0]
		g[0][t] = 1.0
		g[t][0] = 1.0
		l = shortest_path(g, indices=0)
	return g

def define_higher_order_relationships(generator, nodes, num_relationships, max_relationship_size):
	# initialize a dict representing the set of relationships
	relationships = {}
	for n in range(nodes):
		relationships[n] = []
	# ensure that the size of the higher order relationships is compatible with the graph size
	rel_size = min(nodes-1, max_relationship_size)
	# generate relationship targets
	relationship_targets = generator.choice(range(nodes), size=num_relationships)
	# determine the size of each relationship
	relationship_sizes = generator.integers(low=2, high=rel_size, size=num_relationships, endpoint=True)
	# generate arrays representing relationships
	for i in range(num_relationships):
		relationship = np.zeros(nodes)
		# choose which nodes will participate in it, excluding the target node
		participants = generator.choice( [j for j in range(nodes) if j != relationship_targets[i]], size=relationship_sizes[i], replace=False)
		relationship[participants] = 1.0
		# add the new higher order relationship to the dict
		relationships[relationship_targets[i]].append(relationship)
	return relationships

def run_simuation(generator, graph, higher_order_relationships, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, consider_higher_order):
	# initialize the message history of each node, with the unsolicited messages at each time step
	outboxes = generator.binomial(1, emission_probability, size=(nodes,time_steps))
	# for the following timesteps, populate the message history with response messages and messages induced by higher order relationships
	for t in range(1, time_steps):
		for node in range(nodes):
			# messages responding to messages sent by parent nodes in the previous time step
			for parent in graph[:][node].nonzero()[0]:
				incoming_message = outboxes[parent][t-1]
				weight = graph[parent][node]
				outboxes[node][t] += generator.binomial(min(int(incoming_message), inbox_cap), weight)
			# messages induced by the influence of higher order relationships
			outboxes[node][t] += consider_higher_order * send_higher_order_response_message(generator, higher_order_relationships[node], outboxes[:,t-1], higher_order_sensitivity, inbox_cap)
	return outboxes

def send_higher_order_response_message(generator, relationships, incoming_messages, higher_order_sensitivity, inbox_cap):
	higher_order_contribution = 0
	# for each higher order relationship, take the geometric mean of the contributing nodes's messages in the previous time step
	for relationship in relationships:
		contributors = np.nonzero(relationship)[0]
		contributions = [incoming_messages[i] for i in contributors if i != 0]
		geo_mean = np.prod(contributions)**(1/float(len(contributions)))
		higher_order_contribution += geo_mean
	# a 1 corresponds to sending a message of interest, a 0 corresponds to not sending a message of interest
	return generator.binomial(min(int(higher_order_contribution), inbox_cap), higher_order_sensitivity)

def calculate_transfer_entropy(series, history_length=1):
	# calculate the transfer entropy between each pair of time series
	l = list(product(range(len(series)),repeat=2))
	#te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length, condition=[series[i] for i in range(len(series)) if (i!=source and i!=target)]) for source, target in l]
	te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length) for source, target in l]
	# use the transfer entropy values as edge weights to construct an adjacency matrix for a directed graph
	te_graph = np.array(te_values_list).reshape((len(series),len(series)))
	threshold = np.mean(te_graph)
	te_graph = np.where(te_graph >= threshold, te_graph, 0.0)
	return te_graph

def extract_rankings(graph, nodes, alpha=0.1, beta=1.0):
	b = np.ones((nodes, 1)) * beta
	katz_centrality = np.linalg.solve(np.eye(nodes,nodes) - (alpha * graph), b).squeeze()
	norm = np.sign(sum(katz_centrality)) * np.linalg.norm(katz_centrality)
	ranking = dict(zip(range(nodes), katz_centrality/norm))
	ranking = sorted([(ranking[key], key) for key in ranking.keys()], reverse = True)
	return ranking

def extract_rank_correlation(true_rankings, test_rankings):
	# calculate the rank correlation between the centrality scores of the ground truth graph and the te graph 
	true_rank_array = np.zeros(len(true_rankings))
	for centrality, node in true_rankings:
		true_rank_array[node] = centrality
	test_rank_array = np.zeros(len(test_rankings))
	for centrality, node in test_rankings:
		test_rank_array[node] = centrality
	rank_correlation = np.corrcoef(true_rank_array, test_rank_array)
	return rank_correlation[0][1]

def experiment(out_dir, min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap):
	ex_results_p_value = pd.DataFrame()
	ex_results_test_mean = pd.DataFrame()
	ex_results_test_std = pd.DataFrame()
	ex_results_control_mean = pd.DataFrame()
	ex_results_control_std = pd.DataFrame()
	ex_results_df = pd.DataFrame()
	for rels in range(min_rels, max_rels+1):
		print("Relationships = {}".format(rels))
		test_rc_results = pd.DataFrame()
		control_rc_results = pd.DataFrame()
		for nodes in range(min_nodes, max_nodes+1):
			print("Nodes = {}".format(nodes))
			test_rank_correlation, control_rank_correlation = trial(rels, nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap)
			test_rc_results[nodes] = test_rank_correlation
			control_rc_results[nodes] = control_rank_correlation

		# save results
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		test_rc_results.to_json(path_or_buf=out_dir+"/"+str(rels)+"_relationships_test_rank_correlation.json")
		control_rc_results.to_json(path_or_buf=out_dir+"/"+str(rels)+"_relationships_control_rank_correlation.json")

		sample_time_series(out_dir, rels, max_nodes, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap)

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
		results.to_csv(path_or_buf=out_dir+"/"+str(rels)+"_relationships_results.csv")
		print(results)
		#ex_results[rels] = p_values
		ex_results_p_value[rels] = p_values
		ex_results_test_mean[rels] = test_means
		ex_results_test_std[rels] = test_std_devs
		ex_results_control_mean[rels] = control_means
		ex_results_control_std[rels] = control_std_devs
		ex_results_df[rels] = df
	ex_results_p_value.to_csv(path_or_buf=out_dir+"/ex_results_p_value.csv")
	ex_results_test_mean.to_csv(path_or_buf=out_dir+"/ex_results_test_mean.csv")
	ex_results_test_std.to_csv(path_or_buf=out_dir+"/ex_results_test_std.csv")
	ex_results_control_mean.to_csv(path_or_buf=out_dir+"/ex_results_control_mean.csv")
	ex_results_control_std.to_csv(path_or_buf=out_dir+"/ex_results_control_std.csv")
	ex_results_df.to_csv(path_or_buf=out_dir+"/ex_results_df.csv")

def trial(num_relationships, nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap):
	test_rank_correlation = np.zeros(trials)
	control_rank_correlation = np.zeros(trials)

	for t in range(trials):
		generator = np.random.default_rng()
		g = Generate_Watts_Strogatz_Graph(generator, nodes, path_weight, rewire_probability)
		r = define_higher_order_relationships(generator, nodes, num_relationships, max_relationship_size)
		true_ranking = extract_rankings(g, nodes)
		
		test_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, True)
		test_time_series = test_time_series > 0
		test_te = calculate_transfer_entropy(test_time_series)
		test_ranking = extract_rankings(test_te, nodes)
		test_correlation = extract_rank_correlation(true_ranking, test_ranking)
		test_rank_correlation[t] = test_correlation

		control_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, False)
		control_time_series = control_time_series > 0
		control_te = calculate_transfer_entropy(control_time_series)
		control_ranking = extract_rankings(control_te, nodes)
		control_correlation = extract_rank_correlation(true_ranking, control_ranking)
		control_rank_correlation[t] = control_correlation

	return (test_rank_correlation, control_rank_correlation)

def sample_time_series(out_dir, num_relationships, nodes, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap):
	generator = np.random.default_rng()
	g = Generate_Watts_Strogatz_Graph(generator, nodes, path_weight, rewire_probability)
	r = define_higher_order_relationships(generator, nodes, num_relationships, max_relationship_size)
	test_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, True)
	control_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, False)

	test = test_time_series[0]
	bin_test = (test_time_series[0] > 0) * 1
	control = control_time_series[0]
	bin_control = (control_time_series[0] > 0) * 1

	print(test)
	print(bin_test)
	print(control)
	print(bin_control)

	#with open(str(num_relationships)+"_relationships_sample_time_series", "w") as f:
	np.savez(out_dir+"/"+str(num_relationships)+"_relationships_sample_time_series", test=test, bin_test=bin_test, control=control, bin_control=bin_control)

def sample_graphs(num_relationships, nodes, path_weight, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, out_dir):
	for rewire_probability in [0.1, 0.5, 0.9]:
		generator = np.random.default_rng()
		g = Generate_Watts_Strogatz_Graph(generator, nodes, path_weight, rewire_probability)
		r = define_higher_order_relationships(generator, nodes, num_relationships, max_relationship_size)
		name = "sample_graph_MAS_rw_"+str(int(10*rewire_probability))
		title = "Multi Agent System, rewire probability = "+str(rewire_probability)
		save_graph(g, out_dir, name, title)
		
		test_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, True)
		test_time_series = test_time_series > 0
		test_te = calculate_transfer_entropy(test_time_series)
		name = "sample_graph_TE_rw_"+str(int(10*rewire_probability))
		title = "TE Network, rewire probability = "+str(rewire_probability)
		save_graph(test_te, out_dir, name, title)

def save_graph(g, out_dir, name, title):
	print(g)
	graph = nx.DiGraph(g)
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos)
	nx.draw_networkx_edges(graph, pos)
	nx.draw_networkx_labels(graph, pos)
	#plt.show()
	plt.suptitle(title)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	plt.savefig(out_dir+"/"+name+".png")
	plt.close()

def ex_rewire(out_dir, min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap):
	out_dir = "output/test_2/rewire_probability"
	rewire_probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]
	for rewire_probability in rewire_probabilities:
		experiment(out_dir+"_"+str(rewire_probability), min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap)

def ex_max_weight(out_dir, min_rels, max_rels, min_nodes, max_nodes, trials, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap):
	out_dir = "output/test_2/max_path_weight"
	max_path_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
	for max_path_weight in max_path_weights:
		experiment(out_dir+"_"+str(max_path_weight), min_rels, max_rels, min_nodes, max_nodes, trials, max_path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap)

def ex_emission(out_dir, min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, higher_order_sensitivity, inbox_cap):
	out_dir = "output/test_2/emission_probability"
	emission_probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]
	for emission_probability in emission_probabilities:
		experiment(out_dir+"_"+str(emission_probability), min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap)

def ex_hos(out_dir, min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, inbox_cap):
	out_dir = "output/test_2/higher_order_sensitivity"
	higher_order_sensitivities = [0.1, 0.3, 0.5, 0.7, 0.9]
	for higher_order_sensitivity in higher_order_sensitivities:
		experiment(out_dir+"_"+str(higher_order_sensitivity), min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap)

def ex_inbox(out_dir, min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity):
	out_dir = "output/test_2/inbox_cap"
	inbox_caps = [1, 3, 5, 7, 9]
	for inbox_cap in inbox_caps:
		experiment(out_dir+"_"+str(inbox_cap), min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap)

if __name__ == '__main__':
	main()