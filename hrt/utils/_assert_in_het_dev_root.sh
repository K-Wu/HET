# assert pwd is in the hrt/ subdir of the repo
assert_in_het_dev_root() {
    if ! python -c "import utils; assert utils.is_pwd_het_dev_root()"; then
        echo "Error: pwd is not in the hrt/ subdir of the repo"
        exit 1
    fi
}
