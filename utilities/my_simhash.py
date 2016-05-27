import sys
import collections
import time
import csv
import math
import copy
import numpy as np
import pandas as pd
import igraph as gp
import os
#from itertools import izip
#from sklearn.metrics.pairwise import cosine_similarity
import colour
from scipy.spatial.distance import hamming



# Input parsing
if len(sys.argv) != 2:
	print "The input format is not proper ! Please enter in the following format."
	print "python sac1.py <alpha value>"    
	exit(1)
alpha_val_str = sys.argv[1]
alpha_value = float(alpha_val_str)






def main():	
	#print my_hash("CF", 64)
	hash1 = my_simhash([("X", 0.1), ("BC", 0.6)], 8)

	#print hash1
	hash2 = my_simhash([("A", 0.1), ("FG", 0.5)], 8)
	print hash1, hash2
	print sim_function(hash1, hash2)
	#print 8 * hamming([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0])
	



def my_simhash(weighted_features, b):
	result = [0] * b
	for t in [(my_hash(x[0], b), x[1]) for x in weighted_features]:
		bitmask = 0
		print t[0], bin(t[0])
		for i in xrange(b):
			bitmask = 1 << i
			if t[0] & bitmask:
				result[b - i - 1] += t[1]
			else:
				result[b - i - 1] += -t[1]			

	fingerprint = [0] * b			
	for i in xrange(b):
		if result[i] > 0:
			fingerprint[i] = 1
		else:
			fingerprint[i] = 0
	return fingerprint		



# TODO : Change implementation to a better Hash
def my_hash(vertex_edge, b):
	if vertex_edge == "":
		return 0
	else:
		x = ord(vertex_edge[0])<<7
		#m = 10**9 + 7
		m = 100003
		mask = 2**b - 1
		for c in vertex_edge:
			x = ((x*m)^ord(c)) & mask
		x ^= len(vertex_edge)
		if x == -1: 
			x = -2
		return x

def sim_function(u, v):
	return float(1 - hamming(u, v))



# Call the main. Entry point.

if __name__ == "__main__":
	main()
