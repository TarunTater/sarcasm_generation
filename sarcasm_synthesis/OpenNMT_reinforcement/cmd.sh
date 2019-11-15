mkdir model_dir
_dir=model_dir

python train.py -data data/sarcasm -save_model model_dir/sarc_synth -world_size 1 -train_steps 100000 -save_checkpoint_steps 5000 -gpu_ranks 0 
