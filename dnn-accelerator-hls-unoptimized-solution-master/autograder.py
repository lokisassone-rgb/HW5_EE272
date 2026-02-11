import subprocess
import sys
import inspect 
import argparse
import json
import time

CRED = '\033[91m'
CGREEN  = '\33[32m'
CEND = '\033[0m'

def test_rtl_test():
    if "all" == args.layers[0]:
        args.layers = ["./layers/resnet_conv1_params.json", "./layers/resnet_conv2_x_params.json", "./layers/resnet_conv3_1_params.json", "./layers/resnet_conv3_x_params.json", "./layers/resnet_conv4_1_params.json", "./layers/resnet_conv4_x_params.json", "./layers/resnet_conv5_1_params.json", "./layers/resnet_conv5_x_params.json"]
    elif "small" == args.layers[0]:
        args.layers = ["./layers/small_layer1.json", "./layers/small_layer2.json", "./layers/small_layer3.json"]

    run = 0
    passed = 0
    i = 0
    while i < len(args.layers):
        layer = args.layers[i]
        print("Running rtl_test with layer params:", layer)

        if (no_build):
            process = subprocess.run(['make', 'clean_test'], 
                         stdout=subprocess.PIPE if not verbose else None,
                         stderr=subprocess.PIPE if not verbose else None)
        else:
        # when running multiple layers, only build once
            if (i == 0):
                process = subprocess.run(['make', 'clean'], 
                            stdout=subprocess.PIPE if not verbose else None,
                            stderr=subprocess.PIPE if not verbose else None)
            else:
                process = subprocess.run(['make', 'clean_test'], 
                         stdout=subprocess.PIPE if not verbose else None,
                         stderr=subprocess.PIPE if not verbose else None)

        with open(layer) as f:
            data = json.load(f)

        param_str_c = f'''const int IC0 = {data["IC0"]};
const int OC0 = {data["OC0"]};
const int IC1 = {data["IC1"]};
const int OC1 = {data["OC1"]};
const int FX = {data["FX"]};
const int FY = {data["FY"]};
const int OX0 = {data["OX0"]};
const int OY0 = {data["OY0"]};
const int OX1 = {data["OX1"]};
const int OY1 = {data["OY1"]};
const int STRIDE = {data["STRIDE"]}; 
'''

        with open("./src/conv_tb_params.h", "w") as output:
            output.write(param_str_c)

        if (verbose):
            print(f'Layer params: IC0={data["IC0"]} OC0={data["OC0"]} IC1={data["IC1"]} OC1={data["OC1"]} FX={data["FX"]} FY={data["FY"]} OX0={data["OX0"]} OY0={data["OY0"]} OX1={data["OX1"]} OY1={data["OY1"]} STRIDE={data["STRIDE"]}')

        process = subprocess.run(['make'], 
                         stdout=subprocess.PIPE if not verbose else None,
                         stderr=subprocess.PIPE if not verbose else None)

        if process.returncode != 0:
            print(CRED + "Test failed to compile\n" + CEND)
            return 0, 0

        process = subprocess.run(['make', 'rtl_test_no_gui'], 
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)

        if verbose:
            print(process.stdout)

        if process.returncode != 0:
            print(CRED + "Test failed to compile\n" + CEND)
            run += 0
            passed += 0
        else:
            if "Simulation PASSED" in process.stdout:
                print(CGREEN + "Test passed!\n" + CEND)
                run += 1
                passed += 1
            else:
                print(CRED + "Test failed\n" + CEND)
                run += 1
                passed += 0
    
        i += 1
    return run, passed

def test_c_weight_test():
    if "all" == args.layers[0]:
        args.layers = ["./layers/resnet_conv1_params.json", "./layers/resnet_conv2_x_params.json", "./layers/resnet_conv3_1_params.json", "./layers/resnet_conv3_x_params.json", "./layers/resnet_conv4_1_params.json", "./layers/resnet_conv4_x_params.json", "./layers/resnet_conv5_1_params.json", "./layers/resnet_conv5_x_params.json"]
    elif "small" == args.layers[0]:
        args.layers = ["./layers/small_layer1.json", "./layers/small_layer2.json", "./layers/small_layer3.json"]


    run = 0
    passed = 0
    for layer in args.layers:
        print("Running weight c_test with layer params:", layer)

        process = subprocess.run(['make', 'clean'],
                                 stdout=subprocess.PIPE if not verbose else None,
                                 stderr=subprocess.PIPE if not verbose else None)

        with open(layer) as f:
            data = json.load(f)

        compare_filename = layer
        compare_filename = compare_filename.replace("layers", "compare")
        compare_filename = compare_filename.replace("params.json", "weight_gold")
        compare_filename = compare_filename.replace(".json", "_weight_gold")
        compare_filename = "../" + compare_filename

        param_str_c = f'''const int IC0 = {data["IC0"]};
const int OC0 = {data["OC0"]};
const int IC1 = {data["IC1"]};
const int OC1 = {data["OC1"]};
const int FX = {data["FX"]};
const int FY = {data["FY"]};
const int OX0 = {data["OX0"]};
const int OY0 = {data["OY0"]};
const int OX1 = {data["OX1"]};
const int OY1 = {data["OY1"]};
const int STRIDE = {data["STRIDE"]}; 
'''

        with open("./src/conv_tb_params.h", "w") as output:
            output.write(param_str_c)

        if (verbose):
            print(f'Layer params: IC0={data["IC0"]} OC0={data["OC0"]} IC1={data["IC1"]} OC1={data["OC1"]} FX={data["FX"]} FY={data["FY"]} OX0={data["OX0"]} OY0={data["OY0"]} OX1={data["OX1"]} OY1={data["OY1"]} STRIDE={data["STRIDE"]}')

    
        process = subprocess.run(['make', 'weight_c_test', f'''COMPARE_FILE={compare_filename}'''], 
                         stdout=subprocess.PIPE if not verbose else None,
                         stderr=subprocess.PIPE if not verbose else None)


        if process.returncode == 0:
            print(CGREEN + "Test passed!\n" + CEND)
            run += 1
            passed += 1
        else:
            print(CRED + "Test failed\n" + CEND)
            run += 1
            passed += 0

    
    return run, passed

def test_c_input_test():
    if "all" == args.layers[0]:
        args.layers = ["./layers/resnet_conv1_params.json", "./layers/resnet_conv2_x_params.json", "./layers/resnet_conv3_1_params.json", "./layers/resnet_conv3_x_params.json", "./layers/resnet_conv4_1_params.json", "./layers/resnet_conv4_x_params.json", "./layers/resnet_conv5_1_params.json", "./layers/resnet_conv5_x_params.json"]
    elif "small" == args.layers[0]:
        args.layers = ["./layers/small_layer1.json", "./layers/small_layer2.json", "./layers/small_layer3.json"]


    run = 0
    passed = 0
    for layer in args.layers:
        print("Running input c_test with layer params:", layer)

        process = subprocess.run(['make', 'clean'],
                                 stdout=subprocess.PIPE if not verbose else None,
                                 stderr=subprocess.PIPE if not verbose else None)

        with open(layer) as f:
            data = json.load(f)

        compare_filename = layer
        compare_filename = compare_filename.replace("layers", "compare")
        compare_filename = compare_filename.replace("params.json", "input_gold")
        compare_filename = compare_filename.replace(".json", "_input_gold")
        compare_filename = "../" + compare_filename

        param_str_c = f'''const int IC0 = {data["IC0"]};
const int OC0 = {data["OC0"]};
const int IC1 = {data["IC1"]};
const int OC1 = {data["OC1"]};
const int FX = {data["FX"]};
const int FY = {data["FY"]};
const int OX0 = {data["OX0"]};
const int OY0 = {data["OY0"]};
const int OX1 = {data["OX1"]};
const int OY1 = {data["OY1"]};
const int STRIDE = {data["STRIDE"]}; 
'''

        with open("./src/conv_tb_params.h", "w") as output:
            output.write(param_str_c)

        if (verbose):
            print(f'Layer params: IC0={data["IC0"]} OC0={data["OC0"]} IC1={data["IC1"]} OC1={data["OC1"]} FX={data["FX"]} FY={data["FY"]} OX0={data["OX0"]} OY0={data["OY0"]} OX1={data["OX1"]} OY1={data["OY1"]} STRIDE={data["STRIDE"]}')

    
        process = subprocess.run(['make', 'input_c_test', f'''COMPARE_FILE={compare_filename}'''], 
                         stdout=subprocess.PIPE if not verbose else None,
                         stderr=subprocess.PIPE if not verbose else None)


        if process.returncode == 0:
            print(CGREEN + "Test passed!\n" + CEND)
            run += 1
            passed += 1
        else:
            print(CRED + "Test failed\n" + CEND)
            run += 1
            passed += 0

    
    return run, passed

def test_c_fast_test():
    if "all" == args.layers[0]:
        args.layers = ["./layers/resnet_conv1_params.json", "./layers/resnet_conv2_x_params.json", "./layers/resnet_conv3_1_params.json", "./layers/resnet_conv3_x_params.json", "./layers/resnet_conv4_1_params.json", "./layers/resnet_conv4_x_params.json", "./layers/resnet_conv5_1_params.json", "./layers/resnet_conv5_x_params.json"]
    elif "small" == args.layers[0]:
        args.layers = ["./layers/small_layer1.json", "./layers/small_layer2.json", "./layers/small_layer3.json"]


    run = 0
    passed = 0
    for layer in args.layers:
        print("Running c_test with layer params:", layer)

        process = subprocess.run(['make', 'clean'],
                                 stdout=subprocess.PIPE if not verbose else None,
                                 stderr=subprocess.PIPE if not verbose else None)

        with open(layer) as f:
            data = json.load(f)

        param_str_c = f'''const int IC0 = {data["IC0"]};
const int OC0 = {data["OC0"]};
const int IC1 = {data["IC1"]};
const int OC1 = {data["OC1"]};
const int FX = {data["FX"]};
const int FY = {data["FY"]};
const int OX0 = {data["OX0"]};
const int OY0 = {data["OY0"]};
const int OX1 = {data["OX1"]};
const int OY1 = {data["OY1"]};
const int STRIDE = {data["STRIDE"]}; 
'''

        with open("./src/conv_tb_params.h", "w") as output:
            output.write(param_str_c)

        if (verbose):
            print(f'Layer params: IC0={data["IC0"]} OC0={data["OC0"]} IC1={data["IC1"]} OC1={data["OC1"]} FX={data["FX"]} FY={data["FY"]} OX0={data["OX0"]} OY0={data["OY0"]} OX1={data["OX1"]} OY1={data["OY1"]} STRIDE={data["STRIDE"]}')

    
        process = subprocess.run(['make', 'c_fast_test'], 
                         stdout=subprocess.PIPE if not verbose else None,
                         stderr=subprocess.PIPE if not verbose else None)


        if process.returncode == 0:
            print(CGREEN + "Test passed!\n" + CEND)
            run += 1
            passed += 1
        else:
            print(CRED + "Test failed\n" + CEND)
            run += 1
            passed += 0

    
    return run, passed

def test_c_test():
    if "all" == args.layers[0]:
        args.layers = ["./layers/resnet_conv1_params.json", "./layers/resnet_conv2_x_params.json", "./layers/resnet_conv3_1_params.json", "./layers/resnet_conv3_x_params.json", "./layers/resnet_conv4_1_params.json", "./layers/resnet_conv4_x_params.json", "./layers/resnet_conv5_1_params.json", "./layers/resnet_conv5_x_params.json"]
    elif "small" == args.layers[0]:
        args.layers = ["./layers/small_layer1.json", "./layers/small_layer2.json", "./layers/small_layer3.json"]


    run = 0
    passed = 0
    for layer in args.layers:
        print("Running c_test with layer params:", layer)

        process = subprocess.run(['make', 'clean'],
                                 stdout=subprocess.PIPE if not verbose else None,
                                 stderr=subprocess.PIPE if not verbose else None)

        with open(layer) as f:
            data = json.load(f)

        param_str_c = f'''const int IC0 = {data["IC0"]};
const int OC0 = {data["OC0"]};
const int IC1 = {data["IC1"]};
const int OC1 = {data["OC1"]};
const int FX = {data["FX"]};
const int FY = {data["FY"]};
const int OX0 = {data["OX0"]};
const int OY0 = {data["OY0"]};
const int OX1 = {data["OX1"]};
const int OY1 = {data["OY1"]};
const int STRIDE = {data["STRIDE"]}; 
'''

        with open("./src/conv_tb_params.h", "w") as output:
            output.write(param_str_c)

        if (verbose):
            print(f'Layer params: IC0={data["IC0"]} OC0={data["OC0"]} IC1={data["IC1"]} OC1={data["OC1"]} FX={data["FX"]} FY={data["FY"]} OX0={data["OX0"]} OY0={data["OY0"]} OX1={data["OX1"]} OY1={data["OY1"]} STRIDE={data["STRIDE"]}')

    
        process = subprocess.run(['make', 'c_test'], 
                         stdout=subprocess.PIPE if not verbose else None,
                         stderr=subprocess.PIPE if not verbose else None)


        if process.returncode == 0:
            print(CGREEN + "Test passed!\n" + CEND)
            run += 1
            passed += 1
        else:
            print(CRED + "Test failed\n" + CEND)
            run += 1
            passed += 0

    
    return run, passed


parser = argparse.ArgumentParser(description='Autograder for EE272')
parser.add_argument('tests', type=str, nargs='*', default=['all'],
                    help='list of tests you would like to run')
parser.add_argument("-l", "--layers", type=str, nargs='*', help='Layer specification files to run', default=["./layers/small_layer1.json"])
parser.add_argument("--list", action="store_true", help='List all tests')
parser.add_argument("-v", "--verbose", action="store_true", help='Verbose option for printing test output')
parser.add_argument("-n", "--no_build", action="store_true", help='Option for not rebuilding when running RTL tests')

args = parser.parse_args()


verbose = args.verbose
no_build = args.no_build

all_tests = [obj for name,obj in inspect.getmembers(sys.modules[__name__]) 
                        if (inspect.isfunction(obj) and 
                            name.startswith('test') and
                            obj.__module__ == __name__)]


tests = []

if args.list:
    tests_names = [test.__name__ for test in all_tests]
    print(tests_names)
    exit()


if len(args.tests) == 0 or "all" in args.tests:
    tests = all_tests
else:
    for arg in args.tests:
        for test in all_tests:
            if arg in test.__name__:
                tests.append(test)

tests_names = [test.__name__ for test in tests]

print("Tests being run: ", tests_names)

tests_run = 0
tests_passed = 0

start = time.time()

for test in tests:
    run, passed = test()
    tests_run += run
    tests_passed += passed

end = time.time()

print("Tests passed:", tests_passed)
print("Tests run:", tests_run)

print("elapsed time: ", (end-start))
