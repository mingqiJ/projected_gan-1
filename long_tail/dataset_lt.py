import json
import argparse
import numpy as np
import os


def dataset_lt(args):
    pwd = os.path.dirname(args.fn)
    with open(args.fn) as f:
        main_ = json.load(f)

    if main_["labels"]:
        main = np.array(main_["labels"], dtype=object)
    else:
        exit(0)

    img_num_per_cls, classes = get_img_num_per_cls(main, args)
    lt_ds = gen_imbalanced_data(main, img_num_per_cls, classes)
    lt = {"labels": lt_ds.tolist()}

    filename = f"lt_{args.imf}"
    if args.reverse:
        filename += "_reverse"
    with open(os.path.join(pwd, f"{filename}_{args.imf}.json"), "w") as f:
        json.dump(lt, f)


def get_img_num_per_cls(ds, args):
    classes = dict()
    for d in ds:
        if d[1] in classes:
            classes[d[1]] += 1
        else:
            classes[d[1]] = 1
    cls_num = len(classes)
    # img_max = classes[max(classes, key=classes.get)]
    img_max = ds.shape[0] // cls_num

    img_num_per_cls = []
    for cls_idx in set(classes):
        if args.reverse:
            num = img_max * (1/args.imf ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        else:
            num = img_max * (1/args.imf ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))

    return img_num_per_cls, set(classes)


def gen_imbalanced_data(ds, img_num_per_cls, classes):
    new_data = []
    for i, count in zip(classes, img_num_per_cls):
        idx = np.where(ds[:, 1] == i)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:count]
        new_data.append(ds[selec_idx, ...])
    return np.vstack(new_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LongTail Dataset')
    parser.add_argument('--fn', metavar='PATH', type=str, help='path to the dataset json file', default="dataset.json")
    parser.add_argument('--imf', type=int, default=100, help='number of samples in the lt class')
    parser.add_argument('--reverse', action="store_true", help='number of samples in the lt class')
    args = parser.parse_args()

    dataset_lt(args)

