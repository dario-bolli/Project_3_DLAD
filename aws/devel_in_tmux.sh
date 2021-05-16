#!/bin/bash
tmux new-session -d -s dlad -n devel
tmux send-keys -t dlad:devel "source activate pytorch_latest_p37" Enter
tmux send-keys -t dlad:devel "cd ~/code" Enter
#tmux send-keys -t dlad:devel "python tests/test.py --task 1" Enter
#tmux send-keys -t dlad:devel "python tests/test.py --task 2" Enter
#tmux send-keys -t dlad:devel "python tests/test.py --task 4" Enter
#tmux send-keys -t dlad:devel "python tests/test.py --task 5" Enter