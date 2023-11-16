DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${DIR}/_assert_in_het_dev_root.sh"

assert_in_het_dev_root

echo "test_assert_in_het_dev_root.sh passed"