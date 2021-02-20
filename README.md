# Project 2 - Clustering and Drug Discovery
## Due 02/17/2021

![BuildStatus](https://github.com/aseveritt/BMI203_Project2/workflows/HW2/badge.svg?event=push)

In this assignment, you will evaluate results from a high-throughput virtual screen against the SARS-CoV2 Spike protein / Human ACE2 interface.  There are two parts to this assignment and Part 2 requires completion of Part 1. We recommend reading through both Part 1 and Part 2 before beginning this assignment. 

* Part 1 - API and implementation
* Part 2 - Evaluating clustering

The data we are considering comes from [Smith and Smith, 2020](https://chemrxiv.org/articles/preprint/Repurposing_Therapeutics_for_the_Wuhan_Coronavirus_nCov-2019_Supercomputer-Based_Docking_to_the_Viral_S_Protein_and_Human_ACE2_Interface/11871402). In this study, they generated 6 Spike-Ace2 interface poses using MD simulations. They then docked ~10k small molecules against each protein conformation. Provided for you is the top (#1) pose for each ligand docked against one Spike-ACE2 interface conformation, as well as the corresponding SMILES string, AutoDock Vina score, and the “On” bits in the Extended Connectivity Fingerprint for that compound. These can all be found in ligand\_information.csv.


### main
Runs all code in align/\_\_main\_\_.py, useful for part 2
```
python -m algs
```

### testing
Testing is as simple as running
```
python -m pytest test/*
```
from the root directory of this project.





-----------------------------------------------------
# Amanda additions:

- All functions/classes are contained within the script clusters/algs.py
- Script can be called in the following ways:


```
#HIERARCHIAL
from clusters import algs
ligand_dict = algs.read_in_ligands("ligand_information.csv")
hc = algs.HierarchicalClustering(ligand_dict, 
				linkage_metric ="complete",
                                distance_metric = "jaccard", 
                                desired_k = 1).cluster()

plt.figure(figsize =(15, 5))
dn = hierarchy.dendrogram(hc.zmat);
plt.title("Dendrogram of small molecule dataset (dense bits)")
plt.xlabel('Clusters')
plt.ylabel('Jaccard Distance')
plt.axhline(y=3, color='r', linestyle='-')
plt.show()

hc = algs.HierarchicalClustering(ligand_dict, 
				linkage_metric ="complete",
                                distance_metric = "jaccard", 
                                desired_k = 5).cluster()
print(hc.cluster_results)
```


```
#K-MEANS
from clusters import algs
ligand_dict = algs.read_in_ligands("ligand_information.csv")
pc = algs.PartitionClustering(ligand_dict, distance_metric = "jaccard", allow_reinit=True,  
                                desired_k = 5).cluster(seed=3, max_iter = 50)
print(pc.final_clusters)
```



- Script documentation is done through docstrings rendered with Sphinx:
```
open aseveritt/BMI203_Project2/docs/build/html/index.html
```



- Unit testing is all self contained within the script and calls nothing. 

- Plotting and images located in jupyter notebook:
```
open aseveritt/BMI203_Project2/Amanda_Everitt_BMI203_HW2.ipynb
```




