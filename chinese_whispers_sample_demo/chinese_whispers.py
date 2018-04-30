def face_distance(face_encodings, face_to_compare):
	"""
	Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
	for each comparison face. The distance tells you how similar the faces are.
	:param faces: List of face encodings to compare
	:param face_to_compare: A face encoding to compare against
	:return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
	"""
	import numpy as np
	if len(face_encodings) == 0:
		return np.empty((0))

	#print(1/np.linalg.norm(face_encodings - face_to_compare, axis=1))
	#print("****-------------------------*******")
	#print(np.sum(face_encodings*face_to_compare,axis=1))
	#print("face_encodings",face_encodings)
	#print("face_to_compare",face_to_compare)
	return np.sum(face_encodings*face_to_compare,axis=1)



def _chinese_whispers(threshold=0.7, iterations=1):
	#from face_recognition.api import _face_distance
	from random import shuffle
	import networkx as nx
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	# Create graph
	nodes = []
	edges = []

	# test_embeddgings file contains embeddings of 2 unique faces and combinedly 20 embeddings are present (GROUND TRUTH)
	df = pd.read_csv("test_embeddings.csv")
	encodings = np.asarray(df.iloc[:15,:-1])
	image_paths = list(df.iloc[:15,-1])
	#image_paths, encodings = zip(*encoding_list)
	#image_paths = ['A','B','C','D','E','F','G','H','I','J']
	#encodings = (np.asarray([[10,10],[20,10],[30,30],[14,14],[15,51],[11,15],[20,4],[30,9],[4,20],[15,2]])).astype('float')
	
	if len(encodings) <= 1:
		print ("No enough encodings to cluster!")
		return []

	for idx, face_encoding_to_check in enumerate(encodings):
		# Adding node of facial encoding
		node_id = idx + 1

		# Initialize 'cluster' to unique value (cluster of itself)
		node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
		nodes.append(node)
	

		# Facial encodings to compare
		if (idx+1) >= len(encodings):
			# Node is last element, don't create edge
			break
		#print("nodeid, cluster, paths\n",nodes)

		compare_encodings = encodings[idx+1:]

		#print("compare_encodings",compare_encodings)

		distances = face_distance(compare_encodings, face_encoding_to_check)
		

		#print("Distances are ", distances)

		encoding_edges = []
		for i, distance in enumerate(distances):
			if distance > threshold:
				# Add edge if facial match
				edge_id = idx+i+2
				encoding_edges.append((node_id, edge_id, {'weight': distance}))

		#print("Encoding Egdes ", encoding_edges)
		
		edges = edges + encoding_edges
	
	#print("edges", edges)

	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)

	#print(nx.info(G))
	#nx.draw(G,with_labels = True)
	#plt.show()

	# Iterate
	for _ in range(0, iterations):
		cluster_nodes = G.nodes()
		#shuffle(cluster_nodes)
		#print("cluster_nodes", cluster_nodes)
		for node in cluster_nodes:
			#print("Loop cluster nodes")
			neighbors = G[node]
			#print("neighbors",neighbors)
			clusters = {}
			#print("Entered into loop cluster nodes")
			for ne in neighbors:
				#print (ne,G.node[ne]['cluster'])
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
	print(nx.info(G))
	nx.draw(G,with_labels = True)
	plt.show()
	return sorted_clusters



print(_chinese_whispers())














