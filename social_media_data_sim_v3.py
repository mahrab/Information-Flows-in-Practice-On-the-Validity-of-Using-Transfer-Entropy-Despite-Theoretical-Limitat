import argparse
import numpy as np
import time
from itertools import product
import pyinform
import os
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kruskal
from numba import njit
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from seaborn import heatmap

def main():
	parser = initialize_parser()
	args = parser.parse_args()

	start_time = time.time()
	experiment(args.out_dir, args.trials, args.nodes, args.time_steps, args.num_relationships, args.relationship_size, args.P, args.max_Q, args.TE_threshold)
	print("Elapsed time = {} seconds".format(time.time() - start_time))

def initialize_parser():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	
	parser.add_argument('--out_dir', default="output/test_heatmaps_18_DAG3_v3")
	parser.add_argument('--trials', type=int, default=50)
	parser.add_argument('--nodes', type=int, default=100)
	parser.add_argument('--time_steps', type=int, default=8000) # was 2000
	parser.add_argument('--num_relationships', type=float, default=0.1)
	parser.add_argument('--relationship_size', type=int, default=2)
	parser.add_argument('--P', type=float, default=0.5)
	parser.add_argument('--max_Q', type=float, default=0.5)
	parser.add_argument('--TE_threshold', type=float, default=2.0)

	return parser

def generate_tree_graph(generator, nodes, max_Q, branching_factor=3, path_weight=0.9):
	# initialize empty tree graph, and weights for later use
	g = np.zeros((nodes, nodes))
	w = generator.uniform(low=0.01, high=max_Q, size=(nodes, nodes))

	#insert all nodes into the tree, with random parents
	for node in range(1,nodes):
		parents = generator.permutation(range(node))[0:min(node, branching_factor)]
		for parent in parents:
			g[parent][node] = 1.0
	
	# apply weights to tree graph
	g = g*w

	# add a path of high influence
	path = deque([0])
	pos = 0
	while np.sum(g[pos]) > 0:
		child = generator.permutation(g[pos].nonzero()[0])[0]
		g[pos][child] = path_weight
		pos = child
		path.append(child)

	return g, path

def define_higher_order_relationships(generator, nodes, num_relationships, relationship_size):
	relationship_targets = generator.choice(range(nodes), replace=False, size=num_relationships)
	relationship_participants = generator.choice([i for i in range(nodes) if i not in relationship_targets], replace=False, size=(num_relationships, relationship_size))
	return(relationship_targets.copy(), relationship_participants.copy())

@njit
def run_simuation(generator, graph, higher_order_relationships, nodes, time_steps, P, higher_order_sensitivity, consider_higher_order):
	parents = graph.transpose()
	innovation = generator.binomial(1, P, size=(nodes,time_steps))*1.0
	propagation = np.zeros((nodes,time_steps))
	higher_order_propagation = np.zeros((nodes,time_steps))
	for t in range(1, time_steps):
		# propagation
		for child in range(nodes):
			for parent in parents[child].nonzero()[0]:
				weight = parents[child][parent]
				incoming_messages = innovation[parent][t-1] + propagation[parent][t-1] + higher_order_propagation[parent][t-1]
				#propagation[child][t] += generator.binomial(incoming_messages, weight)
				propagation[child][t] += generator.binomial(min(incoming_messages,100), weight)
		# higher order propatation, if enabled
		if consider_higher_order:
			for i in range(len(higher_order_relationships[0])):
				target = higher_order_relationships[0][i]
				participants = higher_order_relationships[1][i]
				contributions = [
					max(innovation[participants[0]][t-1] + propagation[participants[0]][t-1] + higher_order_propagation[participants[0]][t-1], 0.1), 
					max(innovation[participants[1]][t-1] + propagation[participants[1]][t-1] + higher_order_propagation[participants[1]][t-1], 0.1)]
				geo_mean = (contributions[0] * contributions[1])**(0.5)
				higher_order_propagation[target][t] += generator.binomial(geo_mean, higher_order_sensitivity)
	outboxes = innovation + propagation + higher_order_propagation
	return(outboxes.copy())

def calculate_transfer_entropy(series, TE_threshold, history_length=1):
	# calculate the transfer entropy between each pair of time series
	l = list(product(range(len(series)),repeat=2))
	te_values_list = [pyinform.transfer_entropy(series[source], series[target], history_length) for source, target in l]
	# use the transfer entropy values as edge weights to construct an adjacency matrix for a directed graph
	te_graph = np.array(te_values_list).reshape((len(series),len(series)))
	m = np.mean(te_graph)
	s = np.std(te_graph)
	threshold = m + TE_threshold*s
	te_graph = np.where(te_graph >= threshold, te_graph, 0.0)
	return te_graph

@njit
def edit_distance(g1, g2):
	return np.sum(np.absolute((g1 > 0) - (g2 > 0)))

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
	rank_correlation = spearmanr(true_rank_array, test_rank_array).statistic
	return rank_correlation

def trial(num_relationships, nodes, trials, max_Q, relationship_size, time_steps, P, higher_order_sensitivity, TE_threshold):
	test_rank_correlation = np.zeros(trials)
	test_edit_distance = np.zeros(trials)
	control_rank_correlation = np.zeros(trials)
	control_edit_distance = np.zeros(trials)
	gt_graph_size = np.zeros(trials)
	gt_out_degree = np.zeros((trials,nodes))
	test_graph_size = np.zeros(trials)
	control_graph_size = np.zeros(trials)
	test_out_degree = np.zeros((trials,nodes))
	control_out_degree = np.zeros((trials,nodes))

	for t in range(trials):
		print("\rn={}, max_q={}, p={}, trial={}".format(num_relationships, max_Q, P, t), end="")
		generator = np.random.default_rng()
		g, path = generate_tree_graph(generator, nodes, max_Q)
		r = define_higher_order_relationships(generator, nodes, num_relationships, relationship_size)
		true_ranking = extract_rankings(g.copy(), nodes)
		gt_graph_size[t] = np.sum(g > 0)
		gt_out_degree[t] = np.sum(g > 0, axis=1) / np.max(np.sum(g > 0, axis=1))
		
		control_time_series = run_simuation(generator, g.copy(), r, nodes, time_steps, P, higher_order_sensitivity, False)
		bin_control_time_series = control_time_series.copy() > 0
		control_te = calculate_transfer_entropy(bin_control_time_series.copy(), TE_threshold)
		control_distance = edit_distance(g, control_te)
		control_edit_distance[t] = control_distance
		control_graph_size[t] = np.sum(control_te > 0)
		control_out_degree[t] = np.sum(control_te > 0, axis=1) / np.max(np.sum(control_te > 0, axis=1))
		control_ranking = extract_rankings(control_te.copy(), nodes)
		control_correlation = extract_rank_correlation(true_ranking.copy(), control_ranking.copy())
		control_rank_correlation[t] = control_correlation

		test_time_series = run_simuation(generator, g.copy(), r, nodes, time_steps, P, higher_order_sensitivity, True)
		bin_test_time_series = test_time_series > 0
		test_te = calculate_transfer_entropy(bin_test_time_series.copy(), TE_threshold)
		test_distance = edit_distance(g, test_te)
		test_edit_distance[t] = test_distance
		test_graph_size[t] = np.sum(test_te > 0)
		test_out_degree[t] = np.sum(test_te > 0, axis=1) / np.max(np.sum(test_te > 0, axis=1))
		test_ranking = extract_rankings(test_te.copy(), nodes)
		test_correlation = extract_rank_correlation(true_ranking.copy(), test_ranking.copy())
		test_rank_correlation[t] = test_correlation

	return (test_rank_correlation.copy(), control_rank_correlation.copy(), test_edit_distance.copy(), control_edit_distance.copy(), 
		gt_graph_size.copy(), gt_out_degree.copy(), test_graph_size.copy(), control_graph_size.copy(), test_out_degree.copy(), control_out_degree.copy())

def save_graph(g, out_dir, name, title):
	np.savetxt(out_dir+"/"+name+".txt", g, fmt='%.2e')
	graph = nx.DiGraph(g)
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos)
	nx.draw_networkx_edges(graph, pos)
	nx.draw_networkx_labels(graph, pos)
	plt.suptitle(title)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	plt.savefig(out_dir+"/"+name+".png")
	plt.close()

def sample_trial(out_dir, num_relationships, nodes, max_Q, relationship_size, time_steps, P, higher_order_sensitivity, TE_threshold):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	generator = np.random.default_rng()
	g, path = generate_tree_graph(generator, nodes, max_Q)
	r = define_higher_order_relationships(generator, nodes, num_relationships, relationship_size)
	true_ranking = extract_rankings(g, nodes)

	np.savetxt(out_dir+"/higher_order_relationship_targets.txt", r[0], fmt='%.2e')
	np.savetxt(out_dir+"/higher_order_relationship_contributors.txt", r[1], fmt='%.2e')

	name = "sample_graph_Ground Truth"
	title = "Ground Truth Network"
	save_graph(g, out_dir, name, title)

	with open(out_dir+"/true_ranking.txt", "w") as f:
		f.write("Path of High Influence: " + str(path) + "\n")
		for entry in true_ranking:
			f.write("Node "+str(entry[1])+": Katz Centrality = "+str(entry[0])+"\n")
	
	control_time_series = run_simuation(generator, g, r, nodes, time_steps, P, higher_order_sensitivity, False)
	control_time_series = control_time_series > 0
	control_te = calculate_transfer_entropy(control_time_series, TE_threshold)
	control_ranking = extract_rankings(control_te, nodes)
	control_correlation = extract_rank_correlation(true_ranking, control_ranking)

	np.savetxt(out_dir+"/time_series_without_triads.txt", control_time_series, fmt='%.1f')

	name = "sample_graph_TE_without_triads"
	title = "TE Network, Without Higher Order Relationships"
	save_graph(control_te, out_dir, name, title)

	with open(out_dir+"/ranking_without_triads.txt", "w") as f:
		f.write("Rank Correlation = "+str(control_correlation)+"\n")
		for entry in control_ranking:
			f.write("Node "+str(entry[1])+": Katz Centrality = "+str(entry[0])+"\n")

	test_time_series = run_simuation(generator, g, r, nodes, time_steps, P, higher_order_sensitivity, True)
	test_time_series = test_time_series > 0
	test_te = calculate_transfer_entropy(test_time_series, TE_threshold)
	test_ranking = extract_rankings(test_te, nodes)
	test_correlation = extract_rank_correlation(true_ranking, test_ranking)

	np.savetxt(out_dir+"/time_series_with_triads.txt", test_time_series, fmt='%.1f')

	name = "sample_graph_TE_with_triads"
	title = "TE Network, With Higher Order Relationships"
	save_graph(test_te, out_dir, name, title)

	with open(out_dir+"/ranking_with_triads.txt", "w") as f:
		f.write("Rank Correlation = "+str(test_correlation)+"\n")
		for entry in test_ranking:
			f.write("Node "+str(entry[1])+": Katz Centrality = "+str(entry[0])+"\n")

def experiment(o_dir, trials, nodes, time_steps, num_relationships, relationship_size, P, max_Q, TE_threshold):
	# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	# [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
	n_vals = [0.1, 0.2, 0.3]
	q_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	p_vals = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

	#start_time = time.time()
	for n in n_vals:
		rels = int(n*nodes)
		out_dir = o_dir+"/triad_participation_"+str(rels*(1+relationship_size))
	
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		sample_trial(out_dir+"/sample_trial", 1, 10, max_Q, relationship_size, time_steps, P, max_Q, 0.0)

		test_rank_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		control_rank_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		rank_p_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		test_dist_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		control_dist_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		dist_p_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)

		gt_graph_size_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		test_graph_size_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		control_graph_size_results = pd.DataFrame(index=p_vals.copy(), columns=q_vals.copy(), dtype=float)
		
		gt_out_degree_results = pd.DataFrame(index=p_vals.copy(), columns=range(nodes), dtype=float)
		test_out_degree_results = pd.DataFrame(index=p_vals.copy(), columns=range(nodes), dtype=float)
		control_out_degree_results = pd.DataFrame(index=p_vals.copy(), columns=range(nodes), dtype=float)

		for max_q in q_vals:
			for p in p_vals:
				start_time = time.time()
				r = trial(rels, nodes, trials, max_q, relationship_size, time_steps, p, max_q, TE_threshold)

				test_rank_correlation = r[0]
				control_rank_correlation = r[1]
				test_edit_distance = r[2]
				control_edit_distance = r[3]
				gt_graph_size = r[4]
				gt_out_degree = r[5]
				test_graph_size = r[6]
				control_graph_size = r[7]
				test_out_degree = r[8]
				control_out_degree = r[9]

				test_rank_results.loc[p, max_q] = np.mean(test_rank_correlation)
				control_rank_results.loc[p, max_q] = np.mean(control_rank_correlation)
				rank_p_results.loc[p, max_q] = kruskal(test_rank_correlation, control_rank_correlation).pvalue
				test_dist_results.loc[p, max_q] = np.mean(test_edit_distance)
				control_dist_results.loc[p, max_q] = np.mean(control_edit_distance)
				dist_p_results.loc[p, max_q] = kruskal(test_edit_distance, control_edit_distance).pvalue

				gt_graph_size_results.loc[p, max_q] = np.mean(gt_graph_size)
				test_graph_size_results.loc[p, max_q] = np.mean(test_graph_size)
				control_graph_size_results.loc[p, max_q] = np.mean(control_graph_size)

				gt_out_degree_results.loc[p] = np.mean(gt_out_degree, axis=0)
				test_out_degree_results.loc[p] = np.mean(test_out_degree, axis=0)
				control_out_degree_results.loc[p] = np.mean(control_out_degree, axis=0)
				print(", Elapsed time = {} seconds".format(time.time() - start_time))

		test_rank_results.to_csv(path_or_buf=out_dir+"/test_rank_results.csv")
		control_rank_results.to_csv(path_or_buf=out_dir+"/control_rank_results.csv")
		rank_p_results.to_csv(path_or_buf=out_dir+"/rank_p_results.csv")
		test_dist_results.to_csv(path_or_buf=out_dir+"/test_dist_results.csv")
		control_dist_results.to_csv(path_or_buf=out_dir+"/control_dist_results.csv")
		dist_p_results.to_csv(path_or_buf=out_dir+"/dist_p_results.csv")

		gt_graph_size_results.to_csv(path_or_buf=out_dir+"/gt_graph_size_results.csv")
		test_graph_size_results.to_csv(path_or_buf=out_dir+"/test_graph_size_results.csv")
		control_graph_size_results.to_csv(path_or_buf=out_dir+"/control_graph_size_results.csv")
		gt_out_degree_results.to_csv(path_or_buf=out_dir+"/gt_out_degree_results.csv")
		test_out_degree_results.to_csv(path_or_buf=out_dir+"/test_out_degree_results.csv")
		control_out_degree_results.to_csv(path_or_buf=out_dir+"/control_out_degree_results.csv")

		f0 = heatmap(test_rank_results, annot=True, fmt=".2f")
		f0.set(title="Node Rank Correlation With Higher Order Relationships", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/0_"+str(rels*(1+relationship_size))+"_rels_test_rank_results.png")
		plt.close()

		f1 = heatmap(control_rank_results, annot=True, fmt=".2f")
		f1.set(title="Node Rank Correlation Without Higher Order Relationships", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/1_"+str(rels*(1+relationship_size))+"_control_rank_results.png")
		plt.close()

		f2 = heatmap(test_dist_results, annot=True, fmt=".1f")
		f2.set(title="Graph Edit Distance With Higher Order Relationships", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/3_"+str(rels*(1+relationship_size))+"_test_distance_results.png")
		plt.close()

		f3 = heatmap(control_dist_results, annot=True, fmt=".1f")
		f3.set(title="Graph Edit Distance Without Higher Order Relationships", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/4_"+str(rels*(1+relationship_size))+"_control_distance_results.png")
		plt.close()
		
		f4 = heatmap(rank_p_results, annot=True, fmt=".2f")
		f4.set(title="Kruskal-Wallis Test on Node Rank Correlation", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/2_"+str(rels*(1+relationship_size))+"_rank_p_results.png")
		plt.close()

		f5 = heatmap(dist_p_results, annot=True, fmt=".2f")
		f5.set(title="Kruskal-Wallis Test on Graph Edit Distance", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/5_"+str(rels*(1+relationship_size))+"_dist_p_results.png")
		plt.close()

		f6 = heatmap(gt_graph_size_results, annot=True, fmt=".1f")
		f6.set(title="Ground Truth Network Number of Edges", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/6_"+str(rels*(1+relationship_size))+"_gt_graph_size_results.png")
		plt.close()

		f7 = heatmap(test_graph_size_results, annot=True, fmt=".1f")
		f7.set(title="TE Net Number of Edges w/ Higher Order Relationships", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/7_"+str(rels*(1+relationship_size))+"_test_graph_size_results.png")
		plt.close()

		f8 = heatmap(control_graph_size_results, annot=True, fmt=".1f")
		f8.set(title="TE Net Number of Edges w/out Higher Order Relationships", xlabel="Maximum Q", ylabel="P")
		plt.savefig(o_dir+"/8_"+str(rels*(1+relationship_size))+"_control_graph_size_results.png")
		plt.close()

		f9 = heatmap(gt_out_degree_results, annot=False)
		f9.set(title="Ground Truth Network Out Degree", xlabel="Node", ylabel="P")
		plt.savefig(o_dir+"/9_"+str(rels*(1+relationship_size))+"_gt_out_degree_results.png")
		plt.close()

		f10 = heatmap(test_out_degree_results, annot=False)
		f10.set(title="TE Net Out Degree w/ Higher Order Relationships", xlabel="Node", ylabel="P")
		plt.savefig(o_dir+"/10_"+str(rels*(1+relationship_size))+"_test_out_degree_results.png")
		plt.close()

		f11 = heatmap(control_out_degree_results, annot=False)
		f11.set(title="TE Net Out Degree w/out Higher Order Relationships", xlabel="Node", ylabel="P")
		plt.savefig(o_dir+"/11_"+str(rels*(1+relationship_size))+"_control_out_degree_results.png")
		plt.close()

if __name__ == '__main__':
	main()