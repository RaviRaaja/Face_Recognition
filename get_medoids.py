import numpy as np
import pandas as pd
import csv
import sys,os
import argparse
from sklearn.metrics.pairwise import euclidean_distances

def create_medoids(args):
	df_tmp = pd.read_csv(args.input_csv_file)
	df = df_tmp.iloc[:,1:] 

	print("size of dataframe is ", df.shape)
	#Grouping
	labels_existing = list(np.unique(df.iloc[:,0]))
	print("Number of clusters are ", len(labels_existing))

	#Here '1' is name of column containing all the labels from 0 to 150
	groups = df.groupby('1')
	my_list = []

	for i in labels_existing:
		#get the groups i.e. points belonging to one cluster at a time
		tmp = groups.get_group(i)
		# create distance matirx
		distMatrix = euclidean_distances(tmp,tmp)
		# index of min value will result in medoid
		medoid_id = np.argmin(distMatrix.sum(axis=0))
		# append corresponding index of medoid to my_list
		my_list.append(list(tmp.iloc[medoid_id,:]))

	#create the csv file and save all the values of medoids
	with open(args.output_csv_file, "w") as f:
		writer = csv.writer(f)
		writer.writerows(my_list)



def parse_arguments(argv):
	parser = argparse.ArgumentParser(description="Make sure csv is similar to paths_labels_embeddings pattern")
	parser.add_argument('input_csv_file', type=str,
		help='Similar to  paths_labels_embeddings.csv uploaded in repo')
	parser.add_argument('output_csv_file', type=str,
		help='Name of output csv file where embeddings of medoids are saved along labels with extension file_name.csv')
	return parser.parse_args(argv)

if __name__ == '__main__':
	create_medoids(parse_arguments(sys.argv[1:]))