import glob
import os

# this script should be executed from the root directory of this project 
# e.g.,
# cd Mithril
# python ./scripts/run_all_tests.py

if __name__ == "__main__":
    tests = glob.glob("./build/tests/test_*")
    num_tests = len(tests)
    num_passed_tests = 0
    for test in tests:
        print("\n*** Executing test %s ***" % (test))
        if 'mpi' in test:
            retval = os.system('mpirun -np 4 '+test)
        else:
            retval = os.system(test)
        if retval == 0:
            num_passed_tests += 1
    print("\nNumber of passed tests: %s / %s\n" % (num_passed_tests, num_tests))

