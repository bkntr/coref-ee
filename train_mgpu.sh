#!/bin/sh
TRAIN_GPUS=$1
EVAL_GPU=$2
NAME=$3

IFS=',' read -r -a TRAIN_GPUS_ARRAY <<< "$TRAIN_GPUS"
tmux new-session -s "$NAME" -n "ps"  -d "bash"
tmux send-keys -t "$NAME:ps" "GPUS=$TRAIN_GPUS python ps.py $NAME" Enter
I=0
for GPU in ${TRAIN_GPUS_ARRAY[@]}
do
	tmux new-window -n "worker $I" "bash"
	tmux send-keys -t "$NAME:worker $I" "GPUS=$TRAIN_GPUS TASK=$I python worker.py $NAME" Enter
	I=$((I+1))
done
tmux new-window -n "eval" "bash"
tmux send-keys -t "$NAME:eval" "GPU=$EVAL_GPU python continuous_evaluate.py $NAME" Enter
tmux -2 attach-session -d
