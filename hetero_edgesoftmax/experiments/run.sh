#!/bin/bash -x
# TODO: describe the run configurations of baseline, i.e., seastar, graphiler, and our hetero-edgesoftmax here.


# Manual step 1: set up the environment. The following commands assume you are at $(REPO_ROOT)/hetero_edgesoftmax
# The prompt will be something like (dev_dgl_torch) kwu@kwu-machine-name:~/hetero_edgesoftmax$
# Manual step 2: set GPU persistence mode and lock its clock.
# sudo nvidia-smi --persistence-mode=1
# sudo nvidia-smi -lgc 1395,1395

python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 16 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 16 --sparse_format="csr" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 8 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 8 --sparse_format="csr" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 16 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 16 --sparse_format="csr" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 8 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 8 --sparse_format="csr" --sort_by_src 

python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 16 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 16 --sparse_format="coo" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 8 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 128 --num_classes 8 --sparse_format="coo" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 16 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 16 --sparse_format="coo" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 8 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d fb15k --input_dim 32 --num_classes 8 --sparse_format="coo" --sort_by_src 


python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 16 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 16 --sparse_format="csr" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 8 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 8 --sparse_format="csr" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 16 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 16 --sparse_format="csr" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 8 --sparse_format="csr"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 8 --sparse_format="csr" --sort_by_src 

python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 16 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 16 --sparse_format="coo" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 8 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 128 --num_classes 8 --sparse_format="coo" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 16 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 16 --sparse_format="coo" --sort_by_src 
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 8 --sparse_format="coo"
python -m python.RGCN.RGCNSingleLayer -d wikikg2 --input_dim 32 --num_classes 8 --sparse_format="coo" --sort_by_src 