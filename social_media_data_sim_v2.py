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
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

def main():
	parser = initialize_parser()
	args = parser.parse_args()

	start_time = time.time()

	r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 

	mew_rank_corr_results = pd.DataFrame(index=["Without Higher Order Relationships", "With Higher Order Relationships"], columns=r)
	mew_path_results = pd.DataFrame(index=["Without Higher Order Relationships", "With Higher Order Relationships"], columns=r)

	hos_rank_corr_results = pd.DataFrame(index=["Without Higher Order Relationships", "With Higher Order Relationships"], columns=r)
	hos_path_results = pd.DataFrame(index=["Without Higher Order Relationships", "With Higher Order Relationships"], columns=r)

	thr_rank_corr_results = pd.DataFrame(index=["Without Higher Order Relationships", "With Higher Order Relationships"], columns=r)
	thr_path_results = pd.DataFrame(index=["Without Higher Order Relationships", "With Higher Order Relationships"], columns=r)

	for v in r:
		m0 = experiment(
			args.out_dir+"/max_edge_weight_"+str(int(v*10)), args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
			v, args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity, args.inbox_cap, args.tr)
		mew_rank_corr_results.loc["With Higher Order Relationships", v] = m0[1]
		mew_rank_corr_results.loc["Without Higher Order Relationships", v] = m0[2]
		mew_path_results.loc["With Higher Order Relationships",v] = m0[3]
		mew_path_results.loc["Without Higher Order Relationships", v] = m0[4]
		
		m1 = experiment(
			args.out_dir+"/higher_order_sensitivity_"+str(int(v*10)), args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
			args.max_path_weight, args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, v, args.inbox_cap, args.tr)
		hos_rank_corr_results.loc["With Higher Order Relationships", v] = m1[1]
		hos_rank_corr_results.loc["Without Higher Order Relationships", v] = m1[2]
		hos_path_results.loc["With Higher Order Relationships",v] = m1[3]
		hos_path_results.loc["Without Higher Order Relationships", v] = m1[4]

		m2 = experiment(
			args.out_dir+"/TE_filter_threshold_"+str(int(v*10)), args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
			args.max_path_weight, args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity, args.inbox_cap, v)
		thr_rank_corr_results.loc["With Higher Order Relationships", v] = m2[1]
		thr_rank_corr_results.loc["Without Higher Order Relationships", v] = m2[2]
		thr_path_results.loc["With Higher Order Relationships",v] = m2[3]
		thr_path_results.loc["Without Higher Order Relationships", v] = m2[4]

	f0 = mew_rank_corr_results.T.plot(title="Rank Correlation vs Maximum Edge Strength", ylabel="Average Rank Correlation", xlabel="Maximum Edge Weight")
	plt.savefig(args.out_dir+"/f0.png")
	plt.close()
	f1 = mew_path_results.T.plot(title = "Percentage of High Influence Path Recovered vs Maximum Edge Strength", ylabel="Average Path Recovery", xlabel="Maximum Edge Weight")
	plt.savefig(args.out_dir+"/f1.png")
	plt.close()

	f2 = hos_rank_corr_results.T.plot(title="Rank Correlation vs Sensitivity to Higher Order Relationships", ylabel="Average Rank Correlation", xlabel="Sensitivity")
	plt.savefig(args.out_dir+"/f2.png")
	plt.close()
	f3 = hos_path_results.T.plot(title = "Percentage of High Influence Path Recovered vs Sensitivity to Higher Order Relationships", ylabel="Average Path Recovery", xlabel="Sensitivity")
	plt.savefig(args.out_dir+"/f3.png")
	plt.close()

	f4 = thr_rank_corr_results.T.plot(title="Rank Correlation vs Transfer Entropy Network Filtering Threshold", ylabel="Average Rank Correlation", xlabel="Threshold")
	plt.savefig(args.out_dir+"/f4.png")
	plt.close()
	f5 = thr_path_results.T.plot(title = "Percentage of High Influence Path Recovered vs Transfer Entropy Network Filtering Threshold", ylabel="Average Path Recovery", xlabel="Threshold")
	plt.savefig(args.out_dir+"/f5.png")
	plt.close()
	# l = list(product(range(len(series)),repeat=2)) # saving this snippet of code for later use

	'''
	experiment(
		args.out_dir, args.min_relationships, args.max_relationships, args.min_nodes, args.max_nodes, args.trials, 
		args.max_path_weight, args.rewire_probability, args.max_relationship_size, args.time_steps, args.emission_probability, args.higher_order_sensitivity, args.inbox_cap, args.tr)
	'''
	print("Elapsed time = {} seconds".format(time.time() - start_time))


def initialize_parser():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	
	parser.add_argument('--out_dir', default="output/test_trees_8")
	parser.add_argument('--min_relationships', type=int, default=1)
	parser.add_argument('--max_relationships', type=int, default=10)
	parser.add_argument('--max_relationship_size', type=int, default=5)
	parser.add_argument('--min_nodes', type=int, default=10)
	parser.add_argument('--max_nodes', type=int, default=50)
	parser.add_argument('--trials', type=int, default=50)

	parser.add_argument('--time_steps', type=int, default=200) # 200
	parser.add_argument('--inbox_cap', type=int, default=5) # 5
	parser.add_argument('--emission_probability', type=float, default=0.5) # 0.5
	parser.add_argument('--higher_order_sensitivity', type=float, default=0.5) # 0.5
	parser.add_argument('--max_path_weight', type=float, default=0.5) # 0.8
	parser.add_argument('--rewire_probability', type=float, default=0.3) # 0.3
	parser.add_argument('--tr', type=float, default=0.5) # 0.5

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

#@njit
def generate_tree_graph(generator, nodes, max_weight, branching_factor=1, scale_free=0):
	# initialize empty tree graph, and weights for later use
	g = np.zeros((nodes, nodes))
	w = generator.uniform(low=0.01, high=max_weight, size=(nodes, nodes))

	#insert all nodes into the tree, with random parents
	for node in range(1,nodes):
		parents = generator.permutation(range(node))[0:min(node, branching_factor)]
		for parent in parents:
			g[parent][node] = 1.0

	if scale_free > 0:
		#insert additional edges from nodes with high outdegree to nodes with high indegree
		outdegrees = np.sum(g[0:nodes-1], axis=1)
		outdegree_distribution = outdegrees/np.linalg.norm(outdegrees, ord=1)
		for e in range(nodes*scale_free):
			parent = generator.choice(np.arange(nodes-1), p=outdegree_distribution)
			indegrees = np.sum(g[:, parent+1:nodes], axis=0)
			indegree_distribution = indegrees/np.linalg.norm(indegrees, ord=1)
			child = generator.choice(np.arange(parent+1,nodes), p=indegree_distribution)
			g[parent][child] = 1.0

	# apply weights to tree graph
	g = g*w

	# add a path of high influence
	#path = [0]
	path = deque([0])
	pos = 0
	while np.sum(g[pos]) > 0:
		child = generator.permutation(g[pos].nonzero()[0])[0]
		g[pos][child] = 0.9
		pos = child
		path.append(child)


	# return g
	return g, path

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

def calculate_transfer_entropy(series, tr, history_length=1):
	# calculate the transfer entropy between each pair of time series
	l = list(product(range(len(series)),repeat=2))
	#te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length, condition=[series[i] for i in range(len(series)) if (i!=source and i!=target)]) for source, target in l]
	te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length) for source, target in l]
	# use the transfer entropy values as edge weights to construct an adjacency matrix for a directed graph
	te_graph = np.array(te_values_list).reshape((len(series),len(series)))
	#threshold = np.mean(te_graph)
	m = np.mean(te_graph)
	s = np.std(te_graph)
	threshold = m - tr*s
	te_graph = np.where(te_graph >= threshold, te_graph, 0.0)
	return te_graph

def extract_TE_trees(te_graph):
	te_graph = np.triu(te_graph)
	return te_graph

def check_path(te_graph, path):
	path_length = float(len(path)-1)
	TE_path_length = 0.0
	
	#child = path.pop(0)
	child = path.popleft()

	#while path:
	while len(path) > 0:
		parent = child
		#child = path.pop(0)
		child = path.popleft()
		if te_graph[parent][child] > 0:
			TE_path_length += 1.0

	path_presence = TE_path_length/path_length

	return path_presence

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

def experiment(out_dir, min_rels, max_rels, min_nodes, max_nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, tr):
	ex_results_p_value = pd.DataFrame()
	ex_results_test_mean = pd.DataFrame()
	ex_results_test_std = pd.DataFrame()
	ex_results_control_mean = pd.DataFrame()
	ex_results_control_std = pd.DataFrame()
	ex_results_df = pd.DataFrame()
	ex_results_test_path_mean = pd.DataFrame()
	ex_results_test_path_std = pd.DataFrame()
	ex_results_control_path_mean = pd.DataFrame()
	ex_results_control_path_std = pd.DataFrame()

	sample_trial(out_dir+"/sample_trial", 2, 10, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, tr)

	for nodes in range(max_nodes, min_nodes-1, -5):
		print("Nodes = {}".format(nodes))
		test_rc_results = pd.DataFrame()
		test_path_results = pd.DataFrame()
		control_rc_results = pd.DataFrame()
		control_path_results = pd.DataFrame()
		for rels in range(int(min_nodes/5), int(nodes/5)+1):
			print("Relationships = {}".format(rels))
			test_rank_correlation, test_path, control_rank_correlation, control_path = trial(rels, nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, tr)
			test_rc_results[rels] = test_rank_correlation
			test_path_results[rels] = test_path
			control_rc_results[rels] = control_rank_correlation
			control_path_results[rels] = control_path

		# save results
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		test_rc_results.to_json(path_or_buf=out_dir+"/"+str(nodes)+"_nodes_test_rank_correlation.json")
		test_path_results.to_json(path_or_buf=out_dir+"/"+str(nodes)+"_nodes_test_path_presence.json")
		control_rc_results.to_json(path_or_buf=out_dir+"/"+str(nodes)+"_nodes_control_rank_correlation.json")
		control_path_results.to_json(path_or_buf=out_dir+"/"+str(nodes)+"_nodes_control_path_presence.json")

		test_means = test_rc_results.mean()
		test_std_devs = test_rc_results.std()
		control_means = control_rc_results.mean()
		control_std_devs = control_rc_results.std()
		test_path_means = test_path_results.mean()
		test_path_std_devs = test_path_results.std()
		control_path_means = control_path_results.mean()
		control_path_std_devs = control_path_results.std()
		p_values = {}
		df = {}
		for col in test_rc_results:
			p_values[col] = ttest_ind(test_rc_results[col], control_rc_results[col], equal_var=False).pvalue
			df[col] = ttest_ind(test_rc_results[col], control_rc_results[col], equal_var=False).df
		results = pd.DataFrame({"test means": test_means, "test std devs": test_std_devs, "control means": control_means, "control std devs": control_std_devs, 
			"test path means": test_path_means, "test path std devs": test_path_std_devs, "control path means": control_path_means, "control path std devs": control_path_std_devs, 
			"p values": p_values, "dfs": df})
		results.to_csv(path_or_buf=out_dir+"/"+str(nodes)+"_nodes_results.csv")
		print(results)
		ex_results_p_value[nodes] = p_values
		ex_results_test_mean[nodes] = test_means
		ex_results_test_std[nodes] = test_std_devs
		ex_results_control_mean[nodes] = control_means
		ex_results_control_std[nodes] = control_std_devs
		ex_results_df[nodes] = df
		ex_results_test_path_mean[nodes] = test_path_means
		ex_results_test_path_std[nodes] = test_path_std_devs
		ex_results_control_path_mean[nodes] = control_path_means
		ex_results_control_path_std[nodes] = control_path_std_devs
	ex_results_p_value.to_csv(path_or_buf=out_dir+"/ex_results_p_value.csv")
	ex_results_test_mean.to_csv(path_or_buf=out_dir+"/ex_results_test_mean.csv")
	ex_results_test_std.to_csv(path_or_buf=out_dir+"/ex_results_test_std.csv")
	ex_results_control_mean.to_csv(path_or_buf=out_dir+"/ex_results_control_mean.csv")
	ex_results_control_std.to_csv(path_or_buf=out_dir+"/ex_results_control_std.csv")
	ex_results_df.to_csv(path_or_buf=out_dir+"/ex_results_df.csv")
	ex_results_test_path_mean.to_csv(path_or_buf=out_dir+"/ex_results_test_path_mean.csv")
	ex_results_test_path_std.to_csv(path_or_buf=out_dir+"/ex_results_test_path_std.csv")
	ex_results_control_path_mean.to_csv(path_or_buf=out_dir+"/ex_results_control_path_mean.csv")
	ex_results_control_path_std.to_csv(path_or_buf=out_dir+"/ex_results_control_path_std.csv")
	m = process_ex_results(ex_results_p_value, ex_results_df, ex_results_test_mean, ex_results_control_mean, ex_results_test_std, ex_results_control_std,
		ex_results_test_path_mean, ex_results_control_path_mean, ex_results_test_path_std, ex_results_control_path_std)

	return m

def trial(num_relationships, nodes, trials, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, tr):
	test_rank_correlation = np.zeros(trials)
	test_path_detection = np.zeros(trials) # testing this
	control_rank_correlation = np.zeros(trials)
	control_path_detection = np.zeros(trials) # testing this

	for t in range(trials):
		generator = np.random.default_rng()
		#g = Generate_Watts_Strogatz_Graph(generator, nodes, path_weight, rewire_probability)
		#g = generate_tree_graph(generator, nodes, path_weight)
		g, path = generate_tree_graph(generator, nodes, path_weight)
		r = define_higher_order_relationships(generator, nodes, num_relationships, max_relationship_size)
		true_ranking = extract_rankings(g, nodes)
		
		test_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, True)
		test_time_series = test_time_series > 0
		test_te = calculate_transfer_entropy(test_time_series, tr)
		test_te = extract_TE_trees(test_te) # testing this
		test_path_presence = check_path(test_te, path.copy()) # testing this
		test_ranking = extract_rankings(test_te, nodes)
		test_correlation = extract_rank_correlation(true_ranking, test_ranking)
		test_rank_correlation[t] = test_correlation
		test_path_detection[t] = test_path_presence # testing this

		control_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, False)
		control_time_series = control_time_series > 0
		control_te = calculate_transfer_entropy(control_time_series, tr)
		control_te = extract_TE_trees(control_te) # testing this
		control_path_presence = check_path(control_te, path.copy()) # testing this
		control_ranking = extract_rankings(control_te, nodes)
		control_correlation = extract_rank_correlation(true_ranking, control_ranking)
		control_rank_correlation[t] = control_correlation
		control_path_detection[t] = control_path_presence # testing this

	#return (test_rank_correlation, control_rank_correlation)
	return (test_rank_correlation, test_path_detection, control_rank_correlation, control_path_detection)

def save_graph(g, out_dir, name, title):
	#print(g)
	np.savetxt(out_dir+"/"+name+".txt", g, fmt='%.2e')
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

def sample_trial(out_dir, num_relationships, nodes, path_weight, rewire_probability, max_relationship_size, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, tr):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	generator = np.random.default_rng()
	#g = Generate_Watts_Strogatz_Graph(generator, nodes, path_weight, rewire_probability)
	#g = generate_tree_graph(generator, nodes, path_weight)
	g, path = generate_tree_graph(generator, nodes, path_weight) #testing this
	r = define_higher_order_relationships(generator, nodes, num_relationships, max_relationship_size)
	true_ranking = extract_rankings(g, nodes)

	name = "sample_graph_MAS"
	title = "Multi Agent System"
	save_graph(g, out_dir, name, title)

	with open(out_dir+"/true_ranking.txt", "w") as f:
		f.write("Path of High Influence: " + str(path) + "\n") # testing this
		for entry in true_ranking:
			f.write("Node "+str(entry[1])+": Katz Centrality = "+str(entry[0])+"\n")
		
	test_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, True)
	test_time_series = test_time_series > 0
	test_te = calculate_transfer_entropy(test_time_series, tr)
	test_te = extract_TE_trees(test_te) # testing this
	test_path_presence = check_path(test_te, path.copy()) # testing this
	test_ranking = extract_rankings(test_te, nodes)
	test_correlation = extract_rank_correlation(true_ranking, test_ranking)

	np.savetxt(out_dir+"/test_time_series.txt", test_time_series, fmt='%.1f')

	name = "sample_graph_TE_test"
	title = "TE Network, With Higher Order Relationships"
	save_graph(test_te, out_dir, name, title)

	with open(out_dir+"/test_ranking.txt", "w") as f:
		f.write("Path Presence: " + str(test_path_presence) + "\n")
		f.write("Rank Correlation = "+str(test_correlation)+"\n")
		for entry in test_ranking:
			f.write("Node "+str(entry[1])+": Katz Centrality = "+str(entry[0])+"\n")

	control_time_series = run_simuation(generator, g, r, nodes, time_steps, emission_probability, higher_order_sensitivity, inbox_cap, False)
	control_time_series = control_time_series > 0
	control_te = calculate_transfer_entropy(control_time_series, tr)
	control_te = extract_TE_trees(control_te) # testing this
	control_path_presence = check_path(control_te, path.copy()) # testing this
	control_ranking = extract_rankings(control_te, nodes)
	control_correlation = extract_rank_correlation(true_ranking, control_ranking)

	np.savetxt(out_dir+"/control_time_series.txt", control_time_series, fmt='%.1f')

	name = "sample_graph_TE_control"
	title = "TE Network, Without Higher Order Relationships"
	save_graph(control_te, out_dir, name, title)

	with open(out_dir+"/control_ranking.txt", "w") as f:
		f.write("Path Presence: " + str(control_path_presence) + "\n")
		f.write("Rank Correlation = "+str(control_correlation)+"\n")
		for entry in control_ranking:
			f.write("Node "+str(entry[1])+": Katz Centrality = "+str(entry[0])+"\n")

def process_ex_results(
	ex_results_p_value, ex_results_df, 
	ex_results_test_mean, ex_results_control_mean, ex_results_test_std, ex_results_control_std,
	ex_results_test_path_mean, ex_results_control_path_mean, ex_results_test_path_std, ex_results_control_path_std):

	m_ex_results_p_value = ex_results_p_value.mean(axis=None)
	m_ex_results_test_mean = ex_results_test_mean.mean(axis=None)
	m_ex_results_control_mean = ex_results_control_mean.mean(axis=None)
	m_ex_results_test_path_mean = ex_results_test_path_mean.mean(axis=None)
	m_ex_results_control_path_mean = ex_results_control_path_mean.mean(axis=None)

	return (m_ex_results_p_value, m_ex_results_test_mean, m_ex_results_control_mean, m_ex_results_test_path_mean, m_ex_results_control_path_mean)

if __name__ == '__main__':
	main()