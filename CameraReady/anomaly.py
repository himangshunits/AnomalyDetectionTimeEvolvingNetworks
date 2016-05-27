################################################################################################
# PLEASE README!
################################################################################################

# The Code for project 08.Topic-9.Project-8.AnomalyDetection.TimeEvolvingGraphs
# Implemented paper : Web Graph Similarity for Anomaly Detection(Paper 1)
# Implemented Section : 5.5
# CSC 591: Algorithms for Data Guided Business Intelligence.
# Team No : 14
# Memebers:
# Himangshu Ranjan Borah(hborah)
# Rahul Shah(rshah5)
# Krunal Gala(kgala)
# Siddhant Doshi(sadoshi)
# Sushma Ravichandran(sravich)

# Description: The code uses the following Libraries.
# numpy, pandas, igraph
# Installation Instructions: 
# Igraph : pip install python-igraph(Please refer to the official documentations for more platform specific details.)
# The data folder must be present in the directory where this file resides.
# The program generates the Time Series of the similarities between the graphs over time for one dataset at a time.
# Saves the graph in the file called "<dataset_name>_time_series.pdf"
# Saves the series in file called "<dataset_name>_time_series.txt"
# Please run the code as python anomaly.py <dataset_folder> from the directory it resides.
# Sample usage : python anomaly.py datasets/datasets/voices/
# The parameter b for hashing is kept as 64 as a default. Can be increased for more accuracy at the cost of computing efficiency.

# Estimated Running times:
# voices : 0.47 seconds.
# enron : 9.43 seconds.
# p2p : 52 seconds.
# autonomous : 648 seconds.


################################################################################################
################################################################################################

import sys
import time
import math
import copy
import numpy as np
import pandas as pd
import igraph as gp
import os
import matplotlib.pyplot as plt

# The dimension of the hashed mapping.
b = 64

# Input parsing
if len(sys.argv) != 2:
	print "The input format is not proper ! Please enter in the following format."
	print "python anomaly.py <dataset directory containing one dataset>"
	exit(1)
data_dir = sys.argv[1]


# returns the hamming_distance of two binary numbers.
def hamming_distance(hash1, hash2):
	x = (hash1 ^ hash2) & ((1 << b) - 1)
	total = 0
	while x:
		total += 1
		#reset the last non zero bit to 0
		x &= x-1
	return total



def main():
	start_time = time.clock()
	generate_anomaly_series()
	print time.clock() - start_time, "seconds taken to run the code !"

# Function to plot the series of the similarities. It also draws the upper and lower threshold values as horizontle lines.
def series_plot(series, threshold_u, threshold_l):
	panda_series = pd.Series(series)
	plt.plot(panda_series)
	plt.axhline(y=threshold_u, color='r')
	plt.axhline(y=threshold_l, color='g')
	temp_words = data_dir.split("/")
	plt.ylabel('graph_similarity')
	plt.xlabel('time')
	plt.title('Time series of similarilities for ' + temp_words[-2] + 'dataset')
	plt.savefig(temp_words[-2] + "_time_series.pdf")


# Functions to write down the series.
def write_series(series, filename):
	f = open(filename,'w')
	line_to_write = "\n".join([str(x) for x in series])
	f.write(line_to_write)	
	f.close()


# This is the main function that implements the SimHash algorithm as given in 5.5 section of the paper.
# The inputs to the function are a set of tuples which has the identifier of the node or edge in the graph
# and an associated weight with it. It produces and unique hash as described by the algorithm.
def my_simhash(weighted_features):
	answer = [0] * b
	for t in [(my_hash(w[0]), w[1]) for w in weighted_features]:
		my_mask = 0
		for i in xrange(b):
			my_mask = 1 << i
			if t[0] & my_mask:
				answer[b - i - 1] += t[1]
			else:
				answer[b - i - 1] += -t[1]
				
	fp_binary = 0
	for i in xrange(b):
		if answer[i] >= 0:
			fp_binary += 1 << i
	return fp_binary

# An efficient hash function. This function is borrowed from it's original implementation of 
# Charikar's SimHash algorithm for better efficiency. (ref : https://github.com/sangelone/python-hashes)
def my_hash(feature):
	if feature == "":
		return 0
	else:
		x = ord(feature[0])<<7
		#m = 10**9 + 7
		m = 100003
		my_mask = (1<<b) - 1
		for c in feature:
			x = ((x*m)^ord(c)) & my_mask
		x ^= len(feature)
		if x == -1:
			x = -2
		return x

def sim_function(u, v):
	return float(b - hamming_distance(u, v)) / b

# Simmilarity between two graphs.
def graph_similarity(graph1_weighted_features, graph2_weighted_features):
	u = my_simhash(graph1_weighted_features)
	v = my_simhash(graph2_weighted_features)
	return sim_function(u, v)

# Convert graphs to weighted featureslist of tuples. Every node is represented by the string format of it's vertex id.
# Every rdge is represented as "<source_id>_<dest_id>". Vertices are mapped to their Page Rank values for the weight. And Edges 
# are mapped to it's quality according to the eqn. in section 5.3 of the paper.
def convert_to_features(input_graph):
	weighted_features = list()
	#find the page ranks, tune the parameters later.
	page_rank_array = input_graph.pagerank(vertices=None, directed=True, weights=input_graph.es['weight'])
	
	# Do the vertices, only the given vertices or all the rage! TODO
	for vertex in input_graph.vs:
		#print vertex.index
		weighted_features.append((str(vertex.index), page_rank_array[vertex.index]))

	# Do the edges: Sum the weights. Multiply with page rank.
	for edge in input_graph.es:
		u = edge.source
		v = edge.target
		edge_quality = float(page_rank_array[u])/(input_graph.vs[u].outdegree())
		weighted_features.append((str(u) + "_" + str(v), edge_quality))
	return weighted_features


# Find the threshold values for the time Series as given in the project description.
def calc_threshold(result_series):
	count = 0
	for i in xrange(1, len(result_series)):
		count += abs(result_series[i] - result_series[i - 1])
	M = (float)(count)/(len(result_series) - 1)
	pd_series = pd.Series(result_series)
	median = pd_series.median()
	upper = median + 3 * M
	lower = median - 3 * M
	return upper, lower

def generate_anomaly_series():
	#main loop through all the graphs
	graphs_array = list()
	for filename in os.listdir(data_dir):
		current_graph = gp.Graph.Read_Edgelist(data_dir + filename, directed=True)
		#Assuming that the graph is unweighted
		current_graph.es["weight"] = 1
		graphs_array.append(current_graph)


	if(len(graphs_array) == 1):
		print "Can't find series, only one graph !!! Abort."
		exit(0)

	result_series = list()
	# Precompute the features of all the graphs and store it for efficiency.
	graph_weighted_features = [convert_to_features(g) for g in graphs_array]

	for i in xrange(len(graphs_array) - 1):
		print "Current Graph Processing = " + str(i)
		graph_sim = graph_similarity(graph_weighted_features[i], graph_weighted_features[i+1])
		result_series.append(graph_sim)

	print result_series

	[upper, lower] = calc_threshold(result_series)

	print "The Upper and Lower Threshold = " + str(upper) + " :: " + str(lower)

	series_plot(result_series, upper, lower)
	temp_words = data_dir.split("/")
	write_series(result_series, temp_words[-2] + "_time_series.txt")

	print "Graphs plotted and saved !"



# Call the main. Entry point.

if __name__ == "__main__":
	main()