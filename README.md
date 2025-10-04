# KGHRec

**KGHRec**, a knowledge graph–based hypergraph neural network for project–library recommendation.  
---

## Overview

KGHRec combines:
- **Collaborative filtering (CF)** based on project–library interactions, and  
- **Knowledge Graph (KG)** reasoning over library dependencies and project metadata,  
enhanced via a **hypergraph neural network** for higher-order relation modeling.

---

## Environment Requirements

The model has been tested under **Python 3.7.5**, but newer versions are also compatible.

| Dependency | Version (tested) |
|-------------|------------------|
| CUDA | 10.2 |
| PyTorch | 1.11.0 |
| NumPy | 1.21.5 |
| Pandas | 1.3.5 |
| SciPy | 1.4.1 |
| tqdm | 4.64.0 |
| scikit-learn | 0.22 |

To install all dependencies:
```bash
pip install torch==1.11.0 numpy==1.21.5 pandas==1.3.5 scipy==1.4.1 tqdm==4.64.0 scikit-learn==0.22


## Run the Codes

You can run the model with the following command:

python main.py --data_name PyLib/ --path t06/ --use_pretrain 0 --attention 1 --knowledgegraph 1


## Results 

Final results will be stored in subfolders of "result\PyLib\"
