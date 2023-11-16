from .sass_analyzer.visualizer.utils.path_config import trace_pattern


test_paths = [
    "hrt/misc/artifacts/nsys_trace_202307161417/RGAT.aifb.--multiply_among_weights_first_flag.--compact_as_of_node_flag--compact_direct_indexing_flag.32.32.1.nsys-rep",
    "hrt/misc/artifacts/ncu_breakdown_202307082005/HGT.aifb.--multiply_among_weights_first_flag.--compact_as_of_node_flag--compact_direct_indexing_flag.ncu-rep",
    "hrt/misc/artifacts/benchmark_all_202307160037/RGAT.aifb.--multiply_among_weights_first_flag.--compact_as_of_node_flag--compact_direct_indexing_flag.32.64.1.result.log",
]

if __name__ == "__main__":
    import re

    for path in test_paths:
        print(re.match(trace_pattern, path))
