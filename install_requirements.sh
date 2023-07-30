conda install pip
pip install dgl --find-links https://data.dgl.ai/wheels/cu118/repo.html
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ogb nvtx rdflib pandas
pip install chardet
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
