do_all_cases() {
  OUTPUT_DIR=$1
  mkdir -p ${OUTPUT_DIR}
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"
  m=RGAT
  d=fb15k
  c="--compact_as_of_node_flag --compact_direct_indexing_flag"
  mf="--multiply_among_weights_first_flag"
  echo $2 python -m python.$m.train -d $d --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --num_heads 1 $c $mf $3
  eval $2 python -m python.$m.train -d $d --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --num_heads 1 $c $mf $3
}

do_all_cases "misc/artifacts/benchmark_all_$(date +%Y%m%d%H%M)" \
  "" \
  ">>\"\${OUTPUT_DIR}/\$m.64.64.1.\$d..\${c//[[:blank:]]/}.result.log\" 2>&1"
