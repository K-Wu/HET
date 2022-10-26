import numpy as np

if __name__ == "__main__":
    with open("freebase-sout.graph") as fd:
        # each line is formatted as: source dest edge_label
        node_dict = dict()
        edge_label_dict = dict()

        edge_label_edge_dict = dict()
        for line in fd:
            src, dest, edge_label = list(map(int, line.strip().split()))
            if src not in node_dict:
                node_dict[src] = len(node_dict.keys())
            if dest not in node_dict:
                node_dict[dest] = len(node_dict.keys())
            if edge_label not in edge_label_dict:
                edge_label_dict[edge_label] = len(edge_label_dict.keys())
            src_id = node_dict[src]
            dest_id = node_dict[dest]
            edge_label_id = edge_label_dict[edge_label]
            if edge_label_id not in edge_label_edge_dict:
                edge_label_edge_dict[edge_label_id] = list()
            edge_label_edge_dict[edge_label_id].append((src_id, dest_id))
    for key in edge_label_edge_dict:
        np.save(
            "FreebaseExq.edge_label_%d.npy" % edge_label_edge_dict[key],
            np.array(edge_label_edge_dict[key], dtype=np.int32).transpose(),
        )
