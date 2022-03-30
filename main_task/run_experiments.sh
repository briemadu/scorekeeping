#!/bin/sh

# batch 1
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task none -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task none -task TF 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task none -task PS 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task none -task PxTSFS 

python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task rand-reps -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task rand-reps -task TF 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task rand-reps -task PS 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task rand-reps -task PxTSFS 

python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task null-reps -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task null-reps -task TF 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task null-reps -task PS 
python3 main.py -device cuda:0 -bot a -bot_version RL_DIV -control_task null-reps -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task none -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task none -task TF 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task none -task PS 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task none -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task rand-reps -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task rand-reps -task TF 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task rand-reps -task PS 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task rand-reps -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task null-reps -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task null-reps -task TF 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task null-reps -task PS 
python3 main.py -device cuda:1 -bot q -bot_version RL_DIV -control_task null-reps -task PxTSFS 

# batch 2
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task none -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task none -task TF 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task none -task PS 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task none -task PxTSFS 

python3 main.py -device cuda:0 -bot a -bot_version SL -control_task rand-reps -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task rand-reps -task TF 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task rand-reps -task PS 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task rand-reps -task PxTSFS 

python3 main.py -device cuda:0 -bot a -bot_version SL -control_task null-reps -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task null-reps -task TF 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task null-reps -task PS 
python3 main.py -device cuda:0 -bot a -bot_version SL -control_task null-reps -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version SL -control_task none -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task none -task TF 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task none -task PS 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task none -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version SL -control_task rand-reps -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task rand-reps -task TF 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task rand-reps -task PS 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task rand-reps -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version SL -control_task null-reps -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task null-reps -task TF 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task null-reps -task PS 
python3 main.py -device cuda:1 -bot q -bot_version SL -control_task null-reps -task PxTSFS 

# batch 3
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task none -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task none -task TF 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task none -task PS 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task none -task PxTSFS 

python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task rand-reps -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task rand-reps -task TF 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task rand-reps -task PS 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task rand-reps -task PxTSFS 

python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task null-reps -task TFxPS 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task null-reps -task TF 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task null-reps -task PS 
python3 main.py -device cuda:0 -bot a -bot_version ICCV_RL -control_task null-reps -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task none -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task none -task TF 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task none -task PS 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task none -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task rand-reps -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task rand-reps -task TF 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task rand-reps -task PS 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task rand-reps -task PxTSFS 

python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task null-reps -task TFxPS 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task null-reps -task TF 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task null-reps -task PS 
python3 main.py -device cuda:1 -bot q -bot_version ICCV_RL -control_task null-reps -task PxTSFS 
