
import numpy as np
import pandas as pd
import csv, itertools, random, sys, math


##### -----------------------------Functions for Ligands --------------------------------------------------  #####


class Ligand():
	'''Stores relevant information for the Ligand information provided. 
    
    Parameters:
        sequence file (str): FASTA file located in sequence directory to be loaded

    Attributes:
        LigandID (str/int): Identifying Ligand ID (in this project akin to rownames)
        Score (int): value representing any identifying value about ligand; here a AutoDock Vina score
        SMILES (list of integers): Simplified Molecular Input Line Entry 
        denseBits (binary list): vector of bits representing on (1) or off (0) at every location
	'''
	def __init__(self, LigandID, Score, SMILES, denseBits):
		self.ID = LigandID
		self.score = Score
		self.smiles = SMILES
		self.denseBits = denseBits





def read_in_ligands(csvfile, total_bits = 1024):
	'''Read in CSV file listing relevant ligand information.  

	| Convert sparse SMILES scores to dense bit representation. 
    
    Parameters:
        file location (str): CSV file location to be read in. must be comma delimited. 
        total_bits (int): default 1024, will converst short SMILES score to a dense bit score of this length

    Attributes:
		ligand_dict (dict): dictionary where keys are ligand IDs and values are the Ligand class for that particular ID. (input to clustering algos)
	'''

	ligand_dict = {}
	off = [0] * total_bits #empty vector the size of total bits
	with open(csvfile) as fp: 
		csvreader = csv.DictReader(fp)
		for row in csvreader:
			on = [int(x) for x in row["OnBits"].split(",")] #pull vector of "on bits"
			total_bits = [1 if i in on else e for i, e in enumerate(off)] #for length of total off bits -- put 1 where "on" bits are found. 
			final = np.array(total_bits)
			l = Ligand(row["LigandID"], row["Score"], row["SMILES"], final) #generate ligand class
			ligand_dict[row["LigandID"]] = l
	return(ligand_dict)


##### ----------------------------END---------------------------------------------------  #####







##### ---------------------------Helper Functions----------------------------------------------------  #####


def isbinary(a_list):
	'''Checks in numpy array is binary. 
	| 	Returns True if list is binary (only 0/1)

	Parameters:
		a_list (list)
	'''

	a_list = a_list.tolist()
	t = a_list.count(1) + a_list.count(0)
	if (t == len(a_list)): return True
	else: return False

def dist(point1, point2, method="euclidean"):
	'''Calculates the distance between two points. 
	|	Note: "jaccard" and "tanimoto" returns with a scale factor of 10. 

	Parameters:
		point1 (numpy array of numeric values): Vector of integers representing the location of point 1 (can be coordinates or smiles)
		point2 (numpy array of numeric values): Vector of integers representing the location of point 2 (can be coordinates or smiles)
		method (str): default "euclidean". Please use Euclidean for coordinate space and "jaccard" or "tanimoto" for SMILES/binary vectors
	'''

	if method == "euclidean":
		sum_sq = np.sum(np.square(point1 - point2))
		d = np.sqrt(sum_sq)
		return(d)

	elif method == "jaccard" or method == "tanimoto":
		if (isbinary(point1) & isbinary(point2)): #check if input is binary
			intersect = [i == j == 1 for i, j in zip(point1, point2)] #vector of T/F where '1' values match.
			bothAB = float(sum(intersect))
			allA = float(sum(point1)) #sum bc binary 
			allB = float(sum(point2))
		return (bothAB/ (allA + allB - bothAB)) * 10  #random scale factor so my numbers aren't tiny. Completely arbitrary. 
	else:
		print("Not exanded for other options yet"); exit(1)

def linkage(point1, point2, method="single"):
	'''Returns the linkage value between two points depending on linkage metric chosen. 

	Parameters:
		point1 (int): Integer should represent a single cell in proximity matrix
		point2 (int): Integer should represent a single cell in proximity matrix
		method (str): default "single". Other options include "average" and "complete"
	'''

	if method == "single":
		return(min(point1,point2))
	elif method == "average":
		return( (point1 + point2)/2)
	elif method == "complete":
		return(max(point1, point2))
	else:
		print("Not exanded for other options yet"); exit(1)


##### ----------------------------END---------------------------------------------------  #####







##### ---------------------------- Metrics---------------------------------------------------  #####

def adjusted_RandIndex(clustersA, clustersB):
	'''Calculate adjusted Rand Index. 
    
    | This metric compares the similarity between two clusterings, or between your predicted and observed clusterings. 
    | This has been adjusted for the random chance that elements could be grouped together. 

    Parameters:
        clustersA (numpy array): 1D list representing element grouping
        clustersB (numpy array): 1D list representing element grouping

    Attributes:
		rand index (float)
	'''

	#Inspiration from: 
	#https://stackoverflow.com/questions/47733231/computing-adjusted-rand-index
	#https://stats.stackexchange.com/questions/89030/rand-index-calculation

	#clustersA = np.array([0, 0, 1, 1])
	#clustersB = np.array([0, 0, 1, 1])
	c_table =  np.asarray(pd.crosstab(clustersA, clustersB)) #build contingency table

	index = 0
	sum_cols = 0 # ~TP + FP
	sum_rows = 0 # ~TP + FN

	for i in range(len(c_table)):
		curr_row = 0 #this is to hold the col sums
		for j in range(len(c_table)):
			index += n_choose_r(c_table[i][j], 2) #sum per column, choose two
			curr_row += c_table[j][i]
		sum_cols += n_choose_r(sum(c_table[i]), 2)
		sum_rows += n_choose_r(curr_row, 2)

	n = np.sum(c_table) #sum of contingency table
	expected_index = (sum_cols*sum_rows)/n_choose_r(n,2)
	max_index = (sum_cols+sum_rows)/2

	return (index - expected_index)/(max_index-expected_index)


def n_choose_r(n, r):
	'''N choose R (nCr) calculation
    
    Parameters:
        n (int)
        r (int)
	'''
	
	f = math.factorial
	if (n-r)>=0:
		return f(n) // f(r) // f(n-r)
	else:
		return 0


#https://gist.github.com/AlexandreAbraham/5544803
#https://github.com/pgrosjean/Project2/blob/main/clusters/algs.py
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html


def intra_cluster_dist(p_mat, point, friends):
	'''Calculate intra-cluster distances
    
    Parameters:
        p_matrix (pandas dataframe): precomputed distance matrix for all samples included in labels set
        point (integer): index of current sample in proximity matrix
        friends (list): list of indexes of points in same clusters as current point

    Attributes:
    	a (float): mean distance from point to all points in the same cluster
	'''

	if len(friends) == 1: #just itself, no other cluster neighbors to calculate
		return 0
	sub = p_mat.iloc[np.array(friends), point] #distance matrix where column is the point and rows are all within-cluster points
	a = sub.mean(axis=0)
	return(a)


def inter_cluster_dist(p_mat, point, allfoes, labels):
	'''Calculate inter-cluster distances
    
    Parameters:
        p_matrix (pandas dataframe): precomputed distance matrix for all samples included in labels set
        point (integer): index of current sample in proximity matrix
        allfoes (list): list of indexes of points in all other clusters than current point
		labels (numpy array): list representing all datapoints clustering assignment. 

    Attributes:
    	b (float): mean distance from point to all points in the closest cluster. 
	'''

	if len(allfoes) == 0: #just itself, no other cluster neighbors to calculate
		return 0

	sub = p_mat.iloc[np.array(allfoes), point] #distance matrix with all other points from other clusters

	closest_foe = np.argmin(np.array(sub)) #index of the closest point
	index_in_total_list = allfoes[closest_foe] #index of that point in the total label list
	others = np.where(labels == labels[index_in_total_list])[0] #all points in the cluster we've deemed closest. 
	sub = p_mat.iloc[np.array(others), point]
	b = sub.mean(axis=0)
	return(b)

def silhouette_score(proximity_matrix, labels):
	'''Calculate silhouette coefficients for each sample. 
    
    | Note, I made the input a proxmity matrix so we don't have to deal with making sure the distance metrics match again. 


    Parameters:
        proximity_matrix (pandas dataframe): precomputed distance matrix for all samples included in labels set (must be in same order)
        labels (numpy array): list representing all datapoints clustering assignment. 

    Attributes:
		sample silhouette score list (list): score for each element in label set
		total silhouette score (float): mean score for all samples. -- total score for clustering
	'''
	
	#labels = np.array([1,2,3,1,2,4,5,6,3,2,1])
	#(b - a) / max(a, b)
	#a = mean intra-cluster distance
	#b = distance between a sample and the nearest cluster

	score_list = []
	for datapoint in range(len(labels)):
		#return an index of all rows where the cluster identity is the same. 
		#friends will include itself, but it doesnt matter since distance matrix will have NaN
		friends = np.where(labels == labels[datapoint])[0] 
		foes = np.where(labels != labels[datapoint])[0]
		a = intra_cluster_dist(proximity_matrix, datapoint, friends)
		b = inter_cluster_dist(proximity_matrix, datapoint, foes, labels)
		s = (b - a) / max(a, b)
		score_list.append(s)
	return(score_list, np.mean(np.array(score_list)))




def make_distance_matrix(self):
	'''Generate a distance Matrix. 

	| Note: This was originally a sub-function only for the heirarchial class 
	| so it may be written a little funky. Luckily both heirarchial and partion
	| have the same core elements. 
	| 
	| Create a N * N matrix where every row and column represents a Ligand ID. 
	| This represents when the number of clusters isequivalent to the number of 
	| data points. Diagnol values all set to NaN as distance to self should not 
	| be included in future calculations. For all ligand combinations, calculate
	| the distance using the specified distance_metric in self. 

	| Matrix will be symmetrical. 

	'''

	IDnames = list()
	#initialize matrix of inf the size of n * n
	myN = len(self.element_list)
	mat = np.matrix(np.ones((myN, myN)) * np.inf)
	#fill diagnol with Nan (would be 0) since we don't care and it cuts down on calculations
	np.fill_diagonal(mat, np.nan)
	for i in range(0, self.Nclusters):
		IDnames.append(self.element_list[i].ID) #easy way to extract row/column names at the same time. 
		for j in range(0, self.Nclusters):
			#for all combinations of ligarnds
			if mat[i,j] == np.inf: #since matrix is symmetrical, don't want to fill out twice.
				d = dist(self.element_list[i].denseBits, self.element_list[j].denseBits, method=self.distance_metric) #calculate distance between dense bits
				mat[i, j] = d; mat[j,i] = d

	df = pd.DataFrame(data=mat, index=IDnames, columns=IDnames) #change to pd df so finding minimums are easeir. 
	return(df)



##### ----------------------------END---------------------------------------------------  #####










##### ----------------------------Agglomerative---------------------------------------------------  #####

class HierarchicalClustering():
	'''Perform hierarchical/agglomerative clustering.

	| For all values in dictionary, generate a proximity matrix represneting the 
	| distance from all data points to each other. Iteratively find the smallest
	| distance between clusters and merge them together using the specified linkage
	| function until the desiered number of clusters, k, is reached. 

	| Ideally, the number of clusters would not be needed and instead the function would choose based on an ideal 
	| quality metric like the silhoutte width. Or a user could output a dendrogram and select their cut-off distance threshold. 
	| I didn't have time to implement either of those. 

	| SUB functions:
	|	initialize_proximity()
	|	find_closest_clusters()
	|	merge_closest_clusters()
	|	update_link_mat()
	| 	cluster()

    
    Parameters:
        element_dict (dictionary of class Ligand): dictionary where key is identifying element and values is a Ligand class
        linkage_metric (str): ["single" (default), "average", "complete"]
        distance_metric (str): [ "euclidean" (default), "jaccard", "tanimoto" (equivalent to jaccard here)]
        desired_k (int): defaults 1. Desired number of clusters. 
        testing (bool): defaults False. True will not initialize proximity matrix.
	'''


	def __init__(self, element_dict, linkage_metric ="single", distance_metric = "euclidean", desired_k = 1, testing= False):
		self.element_list =  list(element_dict.values())
		self.element_dict =  element_dict
		self.linkage = linkage_metric
		self.Nclusters = len(self.element_list)
		self.distance_metric = distance_metric
		self.k = desired_k
		self.test = testing


	def find_closest_clusters(self):
		'''Find closest clusters in proximity matrix. 

		| Identify the smallest value in proximity matrix and return the column and index 
		| name. Note: order not important since matrix is symmetrical

		Attributes:
			min_col (str): First cluster name with smallest distance to second
			min_row (str): Second cluster name with smallest distance to first 
		'''

		df = self.proximity_matrix
		min_col = df.min().idxmin() #minimum of all columns
		min_row = df[min_col].idxmin() #row where the minimum of that column was observed. 
		#note, order doesn't matter here col + row are symmetrical. 
		return(min_col, min_row)


	def merge_closest_clusters(self, min_row, min_col):
		'''Merge the specified clusters together.

		| Using the specificied linkage function in self, create a new column 
		| (name: '(min_row, min_col)') that represents the new cluster. 
		| This new column will be added to the proximity matrix, while the two 
		| original columns will be removed. 

		| Update self.proximity_matrix with this value. 
		| Update self.Nclusters 
		
		Parameters:
			min_col (str): Name of cluster A
			min_row (str): Name of cluster B
		'''

		df = self.proximity_matrix # easier to read
		new_vals = list() #intialize emtpy vectors for new proximity scores
		
		#col_left = list of all columns that aren't going to be removed
		col_left = df.columns.tolist()
		col_left.remove(min_row); col_left.remove(min_col) 

		for i in col_left:
			d = linkage(df[i][min_col], df[i][min_row], method=self.linkage)
			#for all other columns, calculate the distance between the two columns we're merging and the new column. 
			#d = linkage(ligand_dict[min_col].denseBits, ligand_dict[min_row].denseBits, ligand_dict[i].denseBits, method=self.linkage)
			new_vals.append(d)

		df = df.drop(index=[min_col, min_row], columns=[min_col, min_row]) #remove old columsn from proximity matrix
		
		new_name = "(" + str(min_row) + ", " + str(min_col) + ")" #make new joint name (a,b)
		df[new_name] = new_vals #add new values
		new_vals.append(np.nan)
		df.loc[new_name] = new_vals # add one NaN to represent self-self cell we don't care about

		#print(df)
		self.proximity_matrix = df
		self.Nclusters = self.Nclusters - 1
		return(self)

	def update_link_mat(self, a, b, zindex, merge_indexing_start):
		'''Generate Linkage Matrix. (aka zmat)

		| Matrix is in the form:
		| [Node A, Node B, distance(clusterA, clusterB), Number of original total observations A +B ] 
		| This is an imput to the scipy.dendrogram function so I could do some plotting.  

		Attributes:
			a (str): Name of cluster A
			b (str): Name of cluster B
			zindex (dict): keys = nodes, values = node value for plotting
			merge_indexing_start (int): current node index value
		'''
		val1 = int(zindex[a])  #current node value of A
		val2 = int(zindex[b])  #current node value of B
		
		new_name = "(" + str(a) + ", " + str(b) + ")" #new joint name that matches cluster updating
		zindex[new_name] = merge_indexing_start		  #assign a new value that is incremented from last update
		merge_indexing_start+=1

		obs_number = (a.count(",") + 1) + (b.count(",") + 1) # by counting commas, I know how many leafs are in node. 

		self.zmat = np.vstack([self.zmat, np.array([min(val1, val2), max(val1,val2), self.proximity_matrix[a][b], obs_number])])
		return(zindex, merge_indexing_start)





	def cluster(self):
		'''Cluster input data into specified number of clusters k

		| Initialize proximity matrix, while termination command hasn't been met, 
		| continue to find the closest cluters, merge them, and update the linkage matrix. 
		| Output the final cluster assignments. 

		'''

		#initialization
		if not (self.test): #helpful for when I want to overrite what the proximity mat is for testing
			df = make_distance_matrix(self) 
			self.proximity_matrix = df #one I update
			self.distance_mat = df  #one I do not. 


		#iniitilize dendrogram objects too 
		self.zmat = np.empty((0,4), int) #initialize empty Z matrix  (for dendrogram)
		merge_indexing_start = len(self.element_dict.keys()) #number of leafs + 1
		zindex = {b:a for a, b in enumerate(set(self.element_dict.keys()))} # keys = leaf names, values = incremental from 0 to #leafs

		#evalutation
		while self.Nclusters > self.k:
			a, b = self.find_closest_clusters()
			zindex, merge_indexing_start = self.update_link_mat(a, b, zindex, merge_indexing_start)
			self.merge_closest_clusters(a, b)


		#clean-up
		final_clusters = list(self.proximity_matrix.columns.values) #all column names
		self.cluster_results = [i.replace(")", "").replace("(", "").split(",") for i in final_clusters]   # e.g. [['6'], ['10', ' 8', ' 9'], ['5', ' 4', ' 1', ' 2', ' 0', ' 7']]

		#one nice thing about they way I append columns, is that cluster_results will always be ordered from
		# last clustered -> first clustered
		# e.g. [['6'], ['10', ' 8', ' 9'], ['5', ' 4', ' 1', ' 2', ' 0', ' 7']]
		#so I can just build a single vector at the end w/out keeping track throughout. 
		self.labels = {k:"NA" for k in self.element_dict.keys()}
		index =1
		for i in range(len(self.cluster_results)-1, -1, -1): #start at end
			for element in self.cluster_results[i]:
				self.labels[element.strip()] = index
			index+=1
		#e.g. {'0': 1, '1': 1, '2': 1, '4': 1, '5': 1, '6': 'NA', '7': 1, '8': 2, '9': 2, '10': 2}
		return(self)





##### ----------------------------END---------------------------------------------------  #####








##### -----------------------------Kmeans--------------------------------------------------  #####

class PartitionClustering():
	'''Perform artition clustering (naiive kmeans)

	| For all values in dictionary, assign them to one cluster.  
	| Initialize random centroids. While max iterations haven't been met and 
	| centroids have not reached a consensus, update cluster membership 
	| to closest centroid and move centroid to the mean of cluster points. 

	| Limitations: 
	| 	1. random initialization works very poorly
	| 	2. centroids often end up with no members
	| 	3. jaccard distance metric does not have large enough range so centroids converge too quickly
	| 	4. centroids don't represent real data points. 
	| 	5. In jaccard distance, means have to be artificially binarized with a 0.5 threshold. 

	| SUB functions:
	|	initialize() 
	|	assign_centroids()
	|	update_rule()
	|	update_centroids()
	| 	cluster()

    
    Parameters:
        element_dict (dictionary of class Ligand): dictionary where key is identifying element and values is a Ligand class
        distance_metric (str): [ "euclidean" (default), "jaccard", "tanimoto" (equivalent to jaccard here)]
        initialize_method (str): ["forgy" (default)] method to intialize centroids with
        desired_k (int): defaults 1. Desired number of clusters. 
	'''

	def __init__(self, element_dict, distance_metric = "euclidean", initialize_method = "forgy", desired_k = 1, mean_threshold=0.5, allow_reinit=False):
		self.element_list =  list(element_dict.values()) #repetitive, but easier to have. 
		self.element_dict =  element_dict 
		self.Nclusters = len(self.element_list)
		self.distance_metric = distance_metric
		self.init = initialize_method
		self.k = desired_k
		self.t = mean_threshold
		self.allow_reinit = allow_reinit

	def initialize(self):
		'''Initialize Centroids

		| Randomly selects datapoints to serve as initial k centroids
		| ~Really doesn't work, but didn't have time to code k++ 

		'''
		if self.init == "forgy":
			welp = [i.denseBits for i in random.sample(self.element_list, self.k)] 
			self.centroids = welp
		else:
			print("Not expanded for other options yet"); exit(1)

	def assign_centroids(self):
		'''Assign datapoints to closest centroids

		| For all ligands, calculate the distance to all centroids. 
		| Update ligand to be apart of whichever centroid is closest. 
		
		'''
		for ligand in self.element_dict:
			distances = []
			for centroid in self.centroids:
				d = dist(self.element_dict[ligand].denseBits, centroid, self.distance_metric)
				distances.append(d)
			chosen = distances.index(min(distances)) #index of lowest value
			self.labels[ligand] = chosen
		return(self)


	def update_rule(self, df, method = "binarized_mean"):
		'''Rule for how to create new centroids

		| For all ligands in current cluster (df), calcualte the mean values.
		| This either directly becomes your centroid value (method == "mean")
		| Or you can use a 0.5 thresholds to keep the format binary. 

		| Note: theres is a 0% gaurentee the centroids will update to represent any
		| real value in the dataset (or nature for that matter)
		
		'''

		if method == "binarized_mean":
			means = np.array(df.mean(axis=0))
			means[means < self.t] = 0
			means[means >= self.t] = 1
			return(means)
		elif method == "mean":
			means = np.array(df.mean(axis=0))
			return(means)
		else: print("Not expanded for other options yet"); exit(1)




	def update_centroids(self):
		'''Update Centroids

		| For all centroids,  in current cluster (df), calcualte the mean values.
		| This either directly becomes your centroid value (method == "mean")
		| Or you can use a 0.5 thresholds to keep the format binary. 

		| Note: theres is a 0% gaurentee the centroids will update to represent any
		| real value in the dataset (or nature for that matter)
		
		'''

		new_centroids = []
		for i in range(0, self.k): #for all centroids
			points = []
			for ID, cluster in self.labels.items():
				if cluster == i: points.append(self.element_dict[ID].denseBits) # for all cluster members, append their dense bits value
			
			if len(points) == 0: #If cluster has no data points
				if (self.allow_reinit):
					rando = random.sample(self.element_list, 1)[0] #get another random sample
					means = (np.array(rando.denseBits)).astype("float") 
					self.labels[rando.ID] = i 	# and assign itself to this cluster
					new_centroids.append(means) #make its coordinates the new centroid
				else: 
					new_centroids.append(self.centroids[i]) #centroid remains in the same place. 
			else: 
				#calculate new centroid coordinates (aka means)
				if (self.distance_metric == "euclidean"): means = self.update_rule(pd.DataFrame(data=points), method="mean") #I test with euclidean, so just do means. 
				else: means = self.update_rule(pd.DataFrame(data=points), method="binarized_mean")
				new_centroids.append(means)

		self.centroids = new_centroids
		return(self)


	def can_I_stop(self, old_centroids):
		clusters_with_members = set(self.labels.values())

		if clusters_with_members != self.k:
			#print("cant end-- you have empty members")
			return False
		else: #if all clusters are full 
			if old_centroids == self.centroids: # and the previous iteration == this iteration
				return True
			else:
				return False

	def cluster(self, seed=3, max_iter = 10):
		'''Perform K-means clustering

		| Randomly choose k points to be initial centroids (highly dependent on seed). 
		| Until the max iteration number is reached or centroids stop moving, 
		| assign datapoints to centroids, and update those centroids using the mean value of all cluster members. 
		
		Attributes:
			self (PartitionClustering class)
			seed (int): Random seed to start with
			max_iter (int): maximum number of iterations you want to allow. 
		'''

		np.random.seed(seed)
		self.initialize()
		#print(self.centroids)
		self.labels = {k:0 for k in self.element_dict.keys()} #all values in single cluster to begin with
		index = 1 
		
		while index <= max_iter:
			old_centroids = self.centroids
			self.assign_centroids()
			self.update_centroids()
			if self.can_I_stop(old_centroids):
				break
			index+=1	


		all_members = []
		for i in range(0, self.k):
			members = []
			for ID, cluster in self.labels.items():
				if cluster == i: members.append(ID)
			all_members.append(members)
		self.final_clusters = all_members
		return(self)
			


##### ---------------------------Main Call----------------------------------------------------  #####



if __name__ == "__main__":
	ligand_dict = read_in_ligands("ligand_information.csv")
	ligand_sub  = dict(itertools.islice(ligand_dict.items(), 10)) 
	test1 = HierarchicalClustering(ligand_sub, linkage_metric ="single", distance_metric = "jaccard", desired_k = 1, testing = False).cluster()
	#test1 = HierarchicalClustering(ligand_sub, linkage_metric ="average", distance_metric = "jaccard", desired_k = 1, testing = False).cluster()
	#test1 = HierarchicalClustering(ligand_sub, linkage_metric ="complete", distance_metric = "jaccard", desired_k = 1, testing = False).cluster()
	#test1 = HierarchicalClustering(ligand_sub, linkage_metric ="single", distance_metric = "jaccard", desired_k = 5, testing = False).cluster()
	print(test.cluster_results)


	test1 = PartitionClustering(ligand_sub, distance_metric = "jaccard", desired_k = 5, allow_reinit=True, mean_threshold= 0.5)
	p_mat = make_distance_matrix(test1)
	test.cluster(max_iter=10)
	labels = np.fromiter(test1.labels.values(), dtype=int)
	a, b = silhouette_score(p_mat , labels)
	print(b)



##### ----------------------------END---------------------------------------------------  #####



