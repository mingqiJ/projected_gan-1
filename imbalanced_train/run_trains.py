import os

syns_data = "../syns_data"
syns_dirs = [os.path.join(syns_data, dir) for dir in os.listdir(syns_data) if os.path.isdir(os.path.join(syns_data, dir))]

for directory in syns_dirs:
    syns_json = os.path.join(directory, "syns.json")
    if os.path.exists(syns_json):
        for metric in ["CAS", "ACC"]:
            print(f"Running {syns_json} on {metric} metric ...")
            os.system(f"python train.py "
                      f"--fname ../../data/cifar10/lt_100.json "
                      f"--fname_syns {syns_json} "
                      f"--calc_{metric}")


