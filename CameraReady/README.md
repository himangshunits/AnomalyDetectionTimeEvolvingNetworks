
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