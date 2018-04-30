import pandas as pd
import numpy as np
import sys,os
import argparse
import csv

def face_distance(face_encodings, face_to_compare):
	if len(face_encodings) == 0:
		return np.empty((0))
	return np.sum(face_encodings*face_to_compare,axis=1)


def main(args):
	df_input = pd.read_csv(args.medoid_file, header=None)
	df_input_1 = df_input.iloc[:,1:]
	print("Number of clusters (Medoids) are ", df_input_1.shape[0])
	
	df_val = pd.read_csv(args.val_file)
	df_val_1 = df_val.iloc[:,2:]
	print("Number of facial encodings to be validated is ", df_val_1.shape[0])

	correlation_list = []
	cluster_id = []
	medoids_encodings = np.asarray(df_input_1)
	
	for i in range(df_val_1.shape[0]):
		temp = np.asarray(df_val_1.iloc[i,:])
		correlation_list = face_distance(medoids_encodings,temp)
		
		if correlation_list[np.argmax(correlation_list)] >= 0.74:
			cluster_id.append(np.argmax(correlation_list))
		else:
			cluster_id.append('unknown')

	df_val_file_paths = list(df_val.iloc[:,0])
	#cluster_id = np.asarray(cluster_id)

	final_list = list(zip(df_val_file_paths,cluster_id))	
	
	with open(args.output_file, "w") as f:
		writer = csv.writer(f)
		writer.writerows(final_list)

def arg_parse(argv):
	parser = argparse.ArgumentParser(description="Validation on two csv files (medoids and validation embeddings)")
	parser.add_argument('medoid_file', type=str,
		help='Enter the path of the csv file containing medoid/centroids')
	parser.add_argument('val_file', type=str,
		help='Enter the path of the csv file containing embeddings which has to be validated')
	parser.add_argument('output_file', type=str,
		help='Enter the path of the csv file containing embeddings which has to be validated')
	return parser.parse_args(argv)


if __name__ == '__main__':
	main(arg_parse(sys.argv[1:]))