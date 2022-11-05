### To train first synthetic data shoudl be generated.


`$ python generate_syns.py --gan_path best_model.pkl  --desc cifar10_lt_100`

which saves the `syns.json` inside the `syns_data/<desc>/` folder.


Now on can run the following to train a classifier (e.g. cifar10)

`$python train.py --fname ../data/cifar10/lt_100.json --fname_syns ../projected_gan/syns_data/cifar10_lt_100/syns.json <--calc_ACC> <--add_embed>
`  
where `--calc_ACC` trains over augmented data (real+syns), `--calc_CAS` trains over all sysn data, 
and `--add_embed` adds different embeddings to the real and sysn data (only useful when using `--calc_ACC`).

Note, `--calc_ACC` and `calc_CAS` can not be used at the same time and should be done at separate runs.

To train only on real data, you can simply remove all the flags and it will read from the `--fname` file.
 
