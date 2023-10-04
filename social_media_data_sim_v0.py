import numpy as np
from itertools import product
import pyinform

def main(nodes, time_steps, probability, retention, edge_choice, edge_params):
	series = initialize_time_series(nodes, time_steps)
	graph = initialize_agent_graph(nodes, edge_choice, edge_params)
	series = run_simulation(nodes, graph, series, time_steps, probability, retention)

def initialize_time_series(nodes, time_steps):
	return np.zeros((nodes, time_steps))

def initialize_agent_graph(nodes, edge_choice, edge_params):
	if edge_choice == 'identical':
		weight = edge_params[0]
		graph = np.ones((nodes,nodes)) * weight
		np.fill_diagonal(graph, 0.0)
		return graph
	elif edge_choice == 'path':
		path = edge_params[0]
		l0 = edge_params[1]
		l1 = edge_params[2]
		h0 = edge_params[3]
		h1 = edge_params[4]
		graph = np.random.uniform(l0, l1, (nodes,nodes))
		for edge in path:
			graph[edge[0],edge[1]] = np.random.uniform(h0, h1)
		np.fill_diagonal(graph, 0.0)
		return graph
	elif edge_choice == 'uniform':
		l = edge_params[0]
		h = edge_params[h]
		graph = np.random.uniform(l, h, (nodes,nodes))
		np.fill_diagonal(graph, 0.0)
		return graph
	else:
		raise exception("Invalid choice of edge_choice. Valid choices are 'identical', 'path', or 'uniform'")

def run_simulation(nodes, graph, series, time_steps, probability, retention):
	inboxes = np.zeros((nodes,nodes,retention+1))
	for time_step in range(time_steps):
		inboxes = clear_current_layer(time_step, inboxes, retention)
		for node in range(nodes):
			inboxes = response_messages(node, time_step, inboxes, graph, retention)
			inboxes = original_messages(node, time_step, inboxes, graph, probability, retention)
		series = update_time_series(series, inboxes, retention, time_step)
	print(float(np.sum(series))/float(series.size))
	np.savetxt("synthetic_time_series.csv", series, delimiter=',')
	te_graph = calculate_transfer_entropy(series)

def clear_current_layer(time_step, inboxes, retention):
	c = time_step%(retention+1)
	inboxes[:, :, c].fill(0.0)
	return inboxes

def response_messages(node, time_step, inboxes, graph, retention):
	c = time_step%(retention+1)
	#a = (time_step-1)%(retention+1)
	for sender in range(len(inboxes)):
		#if inboxes[sender, node, a] > 0:
		if np.any(inboxes[sender, node, :]):
			m = np.random.choice([0,1], p=[1.0-graph[sender][node], graph[sender][node]])
			inboxes[node, :, c] += m
	return inboxes

def original_messages(node, time_step, inboxes, graph, probability, retention):
	c = time_step%(retention+1)
	m = np.random.choice([0.0,1.0], p=[1.0-probability, probability])
	inboxes[node, :, c] += m
	return inboxes

def update_time_series(series, inboxes, retention, time_step):
	c = time_step%(retention+1)
	for node in range(len(inboxes)):
		series[node][time_step] += np.sum(inboxes[node, :, c]) > 0
	return series

def calculate_transfer_entropy(series):
	l = list(product(range(len(series)),repeat=2))
	te_values_list = [pyinform.transfer_entropy(series[source], series[target], 1) for source, target in l]
	te_graph = np.array(te_values_list).reshape((len(series),len(series)))
	return te_graph

if __name__ == '__main__':
	main(nodes=6, time_steps=100, probability=0.10, retention=1, edge_choice='path', edge_params=(((0,3), (3,4), (3,5)), 0.01, 0.40, 0.60, 1.00))