mkdir model_dir
_dir=model_dir

python train.py -data data/pos_ind -save_model model_dir/pos_model -world_size 1 -train_steps 100000 -save_checkpoint_steps 5000 -gpu_ranks 0 
