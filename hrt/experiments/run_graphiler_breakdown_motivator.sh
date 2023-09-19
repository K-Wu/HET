# Find the script path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

source "${DIR}/../utils/_assert_in_het_dev_root.sh"
assert_in_het_dev_root

OUTPUT_DIR=misc/artifacts/motivator_graphiler_breakdown_`date +%Y%m%d%H%M`
mkdir -p ${OUTPUT_DIR}
echo ${OUTPUT_DIR}
nsys profile --gpu-metrics-device=all --force-overwrite true -o ${OUTPUT_DIR}/graphiler.mutag_HGT.bg.breakdown python3 ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT.py mutag 64 64
nsys profile --gpu-metrics-device=all --force-overwrite true -o ${OUTPUT_DIR}/graphiler.fb15k_HGT.bg.breakdown python3 ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT.py fb15k 64 64

nsys profile --gpu-metrics-device=all --force-overwrite true -o ${OUTPUT_DIR}/graphiler.mutag_RGAT.bg.breakdown python3 ../third_party/OthersArtifacts/graphiler/examples/RGAT/RGAT.py mutag 64 64
nsys profile --gpu-metrics-device=all --force-overwrite true -o ${OUTPUT_DIR}/graphiler.fb15k_RGAT.bg.breakdown python3 ../third_party/OthersArtifacts/graphiler/examples/RGAT/RGAT.py fb15k 64 64
