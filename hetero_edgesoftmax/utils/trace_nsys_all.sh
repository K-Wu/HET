# Find the script path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

source "${DIR}/_assert_in_het_dev_root.sh"
assert_in_het_dev_root

source "${DIR}/_do_all_cases.sh"
do_all_cases "misc/artifacts/nsys_trace_$(date +%Y%m%d%H%M)" \
  "nsys profile --force-overwrite true -o \"\$OUTPUT_DIR/\$m.\$d.\$mf.\${c//[[:blank:]]/}.\$dimx.\$dimy.1\" " \
  "-e 1 --no_warm_up"
