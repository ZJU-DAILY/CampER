# **CampER: An Effective Framework for Privacy-Aware Deep Entity Resolution**

CampER, an effective framework for privacy-aware deep entity resolution, which achieves promising ER accuracy and privacy protection of different organizations. CampER consists of two phases, i.e., collaborative match-aware representation learning (CMRL) and privacy-aware similarity measurement (PASM). In the first phase, CMRL is proposed to embed the tuples owned by different organizations into a uni-space to be match-aware without any manually-labeled pairs; In the second phase, PASM supports a cryptographic-secure similarity measurement algorithm. In addition, we present an order-preserving
perturbation algorithm to significantly accelerate the matching computation while guaranteeing zero impact on the ER
results.
## Requirements

* Python 3.7
* PyTorch 1.10.1
* CUDA 11.5
* NVIDIA A100 40G GPU
* HuggingFace Transformers 4.9.2 

Please refer to the source code to install all required packages in Python.

## Datasets

We conduct experiments on eight widely-used datasets, including DBLP-ACM, Walmart-Amazon, Amazon-Google, DBLP-Scholar, Fordors-Zagats, DBLP-ACM(Dirty), DBLP-Scholar(Dirty), and Walmart-Amazon(Dirty). Since the datasets are partially labeled in pairs for the standard ER task, many tuples are not included in the ground truth and cannot be evaluated. To be fair, we filter out these tuples not included in the ground truth to form new tables for each dataset. We provide all the reformed datasets. 

## Run Experimental Case

To conduct the CampER for effective and privacy-aware deep entity resolution on DBLP-ACM:

```
python main.py --task "/Structure/DBLP-ACM"
```

The meaning of the flags:

- `--task`: the datasets conducted on. e.g."/Structure/DBLP-ACM"

- `--path1`: the path of dedupliated table of the organization A. e.g. "./dataset/Structure/DBLP-ACM/train1-dedup.txt"

- `--path2`: the path of dedupliated table of the organization B. e.g. "./dataset/Structure/DBLP-ACM/train2-dedup.txt"

- `--dup_path1`: the path of duplicate detection result of the organization A. e.g. "./dataset/Structure/DBLP-ACM/train1-dup-id.txt"

- `--dup_path2`: the path of duplicate detection result of the organization B. e.g. "./dataset/Structure/DBLP-ACM/train2-dup-id.txt"

- `--train_wdup _path1`: the path of original table of the organization A. e.g. "./dataset/Structure/DBLP-ACM/train1.txt"

- `--train_wdup _path2`: the path of original table of the organization B. e.g. "./dataset/Structure/DBLP-ACM/train2.txt"

- `--match_path`: the path of dedupilated matched set. e.g. "./dataset/Structure/DBLP-ACM/match-dedup.txt"

- `--rounds`: total collaborative fine-tuning round. e.g. 30

- `--dp_epsilon`: the privacy budget of each owner to perform a single-round collaborative training. e.g. 0.2


## Acknowledgementt

The original datasets are from [DeepMatcher](https://github.com/anhaidgroup/deepmatcher).
