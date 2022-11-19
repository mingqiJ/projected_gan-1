import dnnlib
import legacy
import os
import numpy as np
import torch
import PIL
import json
import argparse


def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def gen_syns_data(gan_path,
                  dir="syns_data",
                  json_fname="sysn.json",
                  desc="cifar10_lt100",
                  imgs_per_cls=5000,
                  batch_size=16,
                  truncation_psi=1.0,
                  noise_mode="const",
                  random_seed=33):
    """

    :param stylegan_path:
    :param gap_img_num_per_cls:
    :param batch_size:
    :param truncation_psi:
    :param noise_mode:
    :param device:
    :return:
    """
    save_dir = os.path.join(dir, desc)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    with dnnlib.util.open_url(gan_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        G.eval()
        G.requires_grad = False
    assert G.c_dim

    num_classes = G.c_dim
    classes = np.arange(num_classes).tolist()
    sysn_imgs_per_cls = num_classes * [imgs_per_cls]
    syns_list = []

    for class_idx, count in zip(classes, sysn_imgs_per_cls):
        # make outdir
        cur_dir = f"syns_{class_idx:05d}"
        cur_path = os.path.join(save_dir, cur_dir)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)

        rem = count
        idx = 0
        while rem > 0:
            cur_batch_size = batch_size if rem >= batch_size else rem
            rem -= cur_batch_size
            # label
            label = torch.zeros([cur_batch_size, G.c_dim], device=device)
            label[:, class_idx] = 1
            # noise
            z = torch.from_numpy(np.random.randn(cur_batch_size, G.z_dim)).to(device).float()
            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = make_transform((0, 0), 0)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            # generate
            imgs = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # save images
            for img in imgs:
                file_name = f"img_syn{idx:08d}.png"
                PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(os.path.join(save_dir, cur_dir, file_name))
                syns_list.append([os.path.join(cur_dir, file_name), class_idx])
                idx += 1

        # save json file
    with open(os.path.join(save_dir, json_fname), "w") as f:
        json.dump({"labels": syns_list}, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_path', type=str, required=True)
    parser.add_argument('--desc', type=str)

    parser.add_argument('--json_fname', type=str, default="syns.json")
    parser.add_argument('--dir', type=str, default="syns_data")
    parser.add_argument('--imgs_per_cls', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--truncation_psi', type=float, default=1.5)
    parser.add_argument('--random_seed', type=int, default=17)

    args = parser.parse_args()
    if not args.desc:
        args.desc = os.path.dirname(args.gan_path).split('/')[-1]

    with torch.no_grad():
        gen_syns_data(
            gan_path=args.gan_path,
            dir=args.dir,
            json_fname=args.json_fname,
            desc=args.desc,
            imgs_per_cls=args.imgs_per_cls,
            batch_size=args.batch_size,
            truncation_psi=args.truncation_psi,
            random_seed=args.random_seed,
            noise_mode="const",
        )
