""" Face Cluster """
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import os,glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
import csv
def face_distance(face_encodings, face_to_compare):
	import numpy as np
	if len(face_encodings) == 0:
		return np.empty((0))

	#correlation between the face encodings and face given to compare
	return np.sum(face_encodings*face_to_compare,axis=1)

def load_model(model_dir, meta_file, ckpt_file):
	model_dir_exp = os.path.expanduser(model_dir)
	saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
	saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

def _chinese_whispers(encoding_list, threshold=0.7, iterations=50):
	#from face_recognition.api import _face_distance
	from random import shuffle
	import networkx as nx
	# Create graph
	nodes = []
	edges = []

	image_paths, encodings = zip(*encoding_list)

	if len(encodings) <= 1:
		print ("No enough encodings to cluster!")
		return []

	for idx, face_encoding_to_check in enumerate(encodings):
		# Adding node of facial encoding
		node_id = idx+1

		# Initialize 'cluster' to unique value (cluster of itself)
		node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
		nodes.append(node)
		# Every encodings obtained is considered as nodes and if 2 nodes are highly correlated then
		# edge is drawn with weight as correlated result.

		# Facial encodings to compare
		if (idx+1) >= len(encodings):
			# Node is last element, don't create edge
			break
		# compute the correlation between each and every images
		compare_encodings = encodings[idx+1:]
		distances = face_distance(compare_encodings, face_encoding_to_check)
		encoding_edges = []
		for i, distance in enumerate(distances):
			# if images are highly correlated then create edge between the nodes
			if distance > threshold:
				# Add edge if facial match
				edge_id = idx+i+2
				encoding_edges.append((node_id, edge_id, {'weight': distance}))

		edges = edges + encoding_edges

	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	# This is start of clustering algorithm checks for connected components in graph
	# every connected component is a cluster
	# Iterate
	for _ in range(0, iterations):
		cluster_nodes = G.nodes()
		#shuffle(cluster_nodes)
		for node in cluster_nodes:
			#print("Loop cluster nodes")
			neighbors = G[node]
			clusters = {}
			#print("Entered into loop cluster nodes")
			for ne in neighbors:
				if isinstance(ne, int):
					if G.node[ne]['cluster'] in clusters:
						clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
						#clusters[G.node[ne]['cluster']].update(clusters[G.node[ne]['cluster']]+G[node][ne]['weight'])
					else:
						clusters[G.node[ne]['cluster']] = G[node][ne]['weight']
						#clusters[G.node[ne]['cluster']].update(G[node][ne]['weight'])

			# find the class with the highest edge weight sum
			edge_weight_sum = 0
			max_cluster = 0
			#use the max sum of neighbor weights class as current node's class
			for cluster in clusters:
				#print("Loop clusters")
				if clusters[cluster] > edge_weight_sum:
					edge_weight_sum = clusters[cluster]
					max_cluster = cluster

			# set the class of target node to the winning local class
			G.node[node]['cluster'] = max_cluster

	clusters = {}

	# Prepare cluster output
	for (_, data) in G.node.items():
		cluster = data['cluster']
		path = data['path']

		if cluster:
			if cluster not in clusters:
				clusters[cluster] = []
			clusters[cluster].append(path)

	# Sort cluster output
	sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

	return sorted_clusters

def cluster_facial_encodings(facial_encodings):
	""" Cluster facial encodings

		Intended to be an optional switch for different clustering algorithms, as of right now
		only chinese whispers is available.

		Input:
			facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

		Output:
			sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
				sorted by largest cluster to smallest

	"""

	if len(facial_encodings) <= 1:
		print ("Number of facial encodings must be greater than one, can't cluster")
		return []

	# Only use the chinese whispers algorithm for now
	sorted_clusters = _chinese_whispers(facial_encodings.items())
	return sorted_clusters

def compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
					embedding_size,nrof_images,nrof_batches,emb_array,batch_size,paths):
	""" Compute Facial Encodings

		Given a set of images, compute the facial encodings of each face detected in the images and
		return them. If no faces, or more than one face found, return nothing for that image.

		Inputs:
			image_paths: a list of image paths

		Outputs:
			facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

	"""

	for i in range(nrof_batches):
		start_index = i*batch_size
		end_index = min((i+1)*batch_size, nrof_images)
		paths_batch = paths[start_index:end_index]
		images = facenet.load_data(paths_batch, False, False, image_size)
		
		
		feed_dict = { images_placeholder:images, phase_train_placeholder:False }
		emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

	facial_encodings = {}
	for x in range(nrof_images):
		facial_encodings[paths[x]] = emb_array[x,:]


	return facial_encodings


def get_onedir(paths):
	dataset = []
	path_exp = os.path.expanduser(paths)
	if os.path.isdir(path_exp):
		images = os.listdir(path_exp)
		image_paths = [os.path.join(path_exp,img) for img in images]

		for x in image_paths:
			if os.path.getsize(x)>0:
				dataset.append(x)
		
	return dataset


def load__(paths):
	user_images = []
	
	subdir = [os.path.join(paths,i) for i in os.listdir(paths)]
	for _ in subdir:
		images = os.listdir(_)
		for i in images:
			user_images.append(os.path.join(_,i))
	return user_images

def main(args):
	from os.path import join, basename, exists
	from os import makedirs
	import numpy as np
	import shutil
	import sys

	if not exists(args.output):
		makedirs(args.output)

	with tf.Graph().as_default():
		with tf.Session() as sess:
			
			image_paths = load__(args.input)
			#get path of images i dataset

			meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
			
			#load pretrained weights of model and graph file
			print('Metagraph file: %s' % meta_file)
			print('Checkpoint file: %s' % ckpt_file)
			load_model(args.model_dir, meta_file, ckpt_file)
			
			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			
			image_size = images_placeholder.get_shape()[1]
			print("image_size:",image_size)
			embedding_size = embeddings.get_shape()[1]
		
			# Run forward pass to calculate embeddings
			print('Runnning forward pass on images') 

			#batch wise passing images to compute the face encodings
			nrof_images = len(image_paths)
			nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
			emb_array = np.zeros((nrof_images, embedding_size))
			facial_encodings = compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
				embedding_size,nrof_images,nrof_batches,emb_array,args.batch_size,image_paths)
			
			sorted_clusters = cluster_facial_encodings(facial_encodings)
			num_cluster = len(sorted_clusters)
			

			#create clusters only if number of images in cluster is more than 20
			list_of_clusters = []
			for i in range(len(sorted_clusters)):
				if(len(sorted_clusters[i]) >= 20 ):
					list_of_clusters.append(sorted_clusters[i])

			# Copy image files to cluster folders
			for idx, cluster in enumerate(list_of_clusters):
				#save all the cluster
				cluster_dir = join(args.output, str(idx))
				if not exists(cluster_dir):
					makedirs(cluster_dir)
				for path in cluster:
					shutil.copy(path, join(cluster_dir, basename(path)))
					
def parse_args():
	"""Parse input arguments."""
	import argparse
	parser = argparse.ArgumentParser(description='Correlation based Graph Clustering Algorithm')
	parser.add_argument('--model_dir', type=str, help='model dir', required=True)
	parser.add_argument('--batch_size', type=int, help='batch size', required=30)
	parser.add_argument('--input', type=str, help='Input dir of images', required=True)
	parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
	args = parser.parse_args()

	return args

if __name__ == '__main__':
	""" Entry point """
	main(parse_args())
