#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-sonic_vla}"
ROOT="/home/bob/GR00T-WholeBodyControl"
STARVLA="/home/bob/SonicStar/starVLA"
CKPT="/home/bob/SonicStar/starVLA/playground/Checkpoints/sonic_latent_scratch_frozen_vlm/checkpoints/steps_90000_pytorch_model.pt"
PY="/home/bob/anaconda3/envs/sonic-mcp/bin/python"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

pkill -f "gear_sonic/scripts/run_sim_loop.py" 2>/dev/null || true
pkill -f "g1_deploy_onnx_ref" 2>/dev/null || true
pkill -f "deploy.sh --input-type zmq_manager sim" 2>/dev/null || true
pkill -f "examples/SonicLatent/eval_files/run_starvla_inference.py" 2>/dev/null || true
pkill -f "ssh -p 983 -N -L 10092:127.0.0.1:10093" 2>/dev/null || true
sleep 1

tmux new-session -d -s "$SESSION" -n main
tmux split-window -h -t "$SESSION":0
tmux split-window -v -t "$SESSION":0.0
tmux split-window -v -t "$SESSION":0.2
tmux select-layout -t "$SESSION":0 tiled

tmux send-keys -t "$SESSION":0.0 "source /home/bob/anaconda3/etc/profile.d/conda.sh && conda activate sonic-mcp && ssh -p 983 -N -L 10092:127.0.0.1:10093 root@123.57.187.96" C-m

tmux send-keys -t "$SESSION":0.1 "source /home/bob/anaconda3/etc/profile.d/conda.sh && conda activate sonic-mcp && cd $ROOT && PYTHONPATH=\$PWD $PY gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port 5555" C-m

tmux send-keys -t "$SESSION":0.2 "source /home/bob/anaconda3/etc/profile.d/conda.sh && conda activate sonic-mcp && cd $ROOT/gear_sonic_deploy && printf '\n' | CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 bash deploy.sh --input-type zmq_manager sim" C-m

tmux send-keys -t "$SESSION":0.3 "source /home/bob/anaconda3/etc/profile.d/conda.sh && conda activate sonic-mcp && cd $STARVLA && echo 'Waiting for deploy port 5557...' && until ss -ltn | grep -q ':5557 '; do sleep 1; done && echo 'Waiting 8s for deploy Init Done...' && sleep 8 && PYTHONPATH=\$PWD:$ROOT $PY examples/SonicLatent/eval_files/run_starvla_inference.py --ckpt-path $CKPT --host 127.0.0.1 --port 10092 --camera-host localhost --camera-port 5555 --state-zmq-host localhost --state-zmq-port 5557 --action-zmq-host localhost --action-zmq-port 5556 --prompt 'pick up the cylinder and throw it into the trash bin' --rate 1.0" C-m

tmux select-pane -t "$SESSION":0.1
tmux attach -t "$SESSION"
