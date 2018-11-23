PREVIOUS_HEAD=$(git rev-parse --abbrev-ref HEAD)

if [[ "${PREVIOUS_HEAD}" = HEAD ]]; then
    echo "you are in a detached HEAD state, please checkout to a brabch"
    return 1
fi

# make sure local master and upstream master are identical
echo "fetching upstream/master..."
git fetch upstream
git checkout master >/dev/null 2>&1

DIFF_WITH_UPSTREAM_MASTER=$(git diff upstream/master)

if [[ -n  ${DIFF_WITH_UPSTREAM_MASTER} ]]; then
    echo "merge your local master branch with upstream/master before running
    this script"
    git checkout "${PREVIOUS_HEAD}"
    return 1
fi

printf "...ok\n\n"


echo "parsing the diff from each commit since master..."
COMMITS_SINCE_MASTER=$(git log master..basic --pretty=format:"%H")
n_deleted_tests=0
for commit in $COMMITS_SINCE_MASTER; do
    git checkout "${commit}" >/dev/null 2>&1
    # grep command find lines matching a test definition deleted in the diff
    # sed command extract the name of the test
    deleted_tests=$(git diff HEAD~1 tests/cloudpickle_test.py | \
        grep -e '^[-]\+\s\+def\stest\_' | \
        sed -n "s/^-\s\+def\s\(.*\).*(.*)\:$/\1/p")

    if [[ -n  ${deleted_tests} ]]; then
        for test in ${deleted_tests}; do
            let n_deleted_tests++
            echo "commit  ${commit:0:7}  deleted ${test}"
        done
    fi
done

printf "...ok\n\n"


# make sure the number of deleted tests the number of current tests
# add up to the previous number of tests in master
echo "counting the tests, and verifying all test were referenced in the last
step or still exist in the branch basic..."

git checkout basic >/dev/null 2>&1
n_tests_in_basic=$(grep -c 'def\stest\_' tests/cloudpickle_test.py)
git checkout master >/dev/null 2>&1
n_tests_in_master=$(grep -c 'def\stest\_' tests/cloudpickle_test.py)
git checkout basic >/dev/null 2>&1

summary_string="
SUMMARY:
-------
number of deleted tests: ${n_deleted_tests}
number of tests in master: ${n_tests_in_master}
number of tests in basic: ${n_tests_in_basic}
"

if [[ ${n_tests_in_master} != $((n_tests_in_basic + n_deleted_tests)) ]]; then
    echo "the test count went wrong: the number of deleted tests + the number
    of current tests in basic does not equal the previous number of tests in
    master"
    echo "${summary_string}"

    git checkout "${PREVIOUS_HEAD}" >/dev/null 2>&1
    return 1
fi

printf "...ok\n\n"

echo "${summary_string}"

printf "\n\n"

echo "making sure the tests summary file is up to date..."
# make sure all deleted tests are mentionned in the summary file
deleted_tests=$(git diff master..basic tests/cloudpickle_test.py | \
    grep -e '^[-]\+\s\+def\stest\_' | \
    sed -n "s/^-\s\+def\s\(.*\).*(.*)\:$/\1/p")

n_warnings=0
for test in ${deleted_tests}; do
    is_in_removed_test_summary_file=$(grep "${test}" removed_tests)
    # test_cm and test_sm are two methods of an object in the test file, they
    # are not actual tests, so we don't include them in the warnings
    if [ "${test}" = "test_cm" ] || [ "${test}" = test_sm ] ; then
        :
    else
        if [[ -z ${is_in_removed_test_summary_file} ]]; then
            let n_warnings++
            echo "WARNING: ${test} is not mentionned in the summary file"
        fi
    fi
done

if [[ $n_warnings = 0 ]]; then
        printf "...ok\n\n"
else
    echo "summary file is not complete (${n_warnings} tests omitted)"
fi

git checkout "${PREVIOUS_HEAD}" >/dev/null 2>&1
