from clusters import algs
import pytest
import pandas as pd
import numpy as np

def test_distances():
	a = algs.dist(np.asarray([1,0,1,1,0,1]),  np.asarray([1,1,0,1,0,0]), method = "tanimoto")
	assert a/10 == 0.4, "Failing to calculate tanimoto index correctly"

	a = algs.dist(np.asarray([1,0,1,1,1,0,0,0]), np.asarray([1,1,0,1,1,0,1,0]), method = "jaccard") #jaccard == tanimoto
	assert a/10 == 0.5, "Failing to calculate tanimoto index correctly"

	a = algs.dist(np.array([10, 2]), np.array([1, 2]), method = "euclidean")
	assert a == 9.0, "Failing to calculate Euclidean wrong"


def test_metrics():
	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
	#using as ground truth
	a = algs.adjusted_RandIndex(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
	assert a == 1, "Failing to ari -- identical"

	a = algs.adjusted_RandIndex(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 2]))
	assert round(a,2) == 0.57, "Failing to ari -- one off"

	a = algs.adjusted_RandIndex(np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3]))
	assert a == 0.0, "Failing to ari"


	ctrl = pd.DataFrame({'a': [np.nan, 17, 21, 31, 23],
						 'b': [17, np.nan, 30, 34, 21],
						 'c': [21, 30, np.nan, 28, 39],
						 'd': [31, 34, 28, np.nan, 43],
						 'e': [23, 21, 39, 43, np.nan]}, 
						 index=["a","b","c","d","e"])
	labels = np.array([3, 3, 1, 1, 2])
	full_list, final_score = algs.silhouette_score(ctrl, labels)
	assert round(final_score, 2) == 0.32, "Failing to calculate silhoutte coef correctly"


def test_partitioning():
	X = {"a": algs.Ligand("a", 0, "whatever", np.array([1, 2])), 
		 "b": algs.Ligand("b", 0, "whatever", np.array([1, 4])),
		 "c": algs.Ligand("c", 0, "whatever", np.array([1, 0])),
		 "d": algs.Ligand("d", 0, "whatever", np.array([10, 2])), 
		 "e": algs.Ligand("e", 0, "whatever", np.array([10, 4])),
		 "f": algs.Ligand("f", 0, "whatever", np.array([10, 0]))
		 }
	tester = algs.PartitionClustering(X, distance_metric = "euclidean", desired_k = 2).cluster()
	assert tester.labels == {'a': 0, 'b': 0, 'c': 0, 'd': 1, 'e': 1, 'f': 1}, "Failing kmeans easy test"

def test_hierarchical():
	ligand_dict = algs.read_in_ligands("ligand_information.csv")
	ctrl = pd.DataFrame({'a': [np.nan, 17, 21, 31, 23],
						 'b': [17, np.nan, 30, 34, 21],
						 'c': [21, 30, np.nan, 28, 39],
						 'd': [31, 34, 28, np.nan, 43],
						 'e': [23, 21, 39, 43, np.nan]}, 
						 index=["a","b","c","d","e"])
	ctr1_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
	
	tester = algs.HierarchicalClustering(ligand_dict, linkage_metric ="single", distance_metric = "euclidean", desired_k = 1, testing=True) #intialiize object
	tester.proximity_matrix = ctrl; tester.Nclusters = 5  #manually overwrite the areas I want to test
	tester.element_dict = ctr1_dict
	tmp = tester.cluster()
	correct_one = [['d', ' e', ' c', ' a', ' b']]
	assert tmp.cluster_results == correct_one, "Failing to merge columns correctly -- one column"

	tester = algs.HierarchicalClustering(ligand_dict, linkage_metric ="single", distance_metric = "euclidean", desired_k = 3, testing=True) #intialiize object
	tester.proximity_matrix = ctrl; tester.Nclusters = 5  #overwrite the areas I want to test
	tester.element_dict = ctr1_dict
	tmp = tester.cluster()
	correct_three = [['d'], ['e'], ['c', ' a', ' b']]
	assert tmp.cluster_results == correct_three, "Failing to merge columns correctly -- three column"



