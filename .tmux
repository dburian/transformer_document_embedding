#!/bin/sh

if tmux has-session "diploma_thesis" 2>/dev/null; then
  tmux attach -t "diploma_thesis"
  exit
fi

tmux new-session -d -s "diploma_thesis" -n "notes" -x $(tput cols) -y $(tput lines)


tmux send-keys -t diploma_thesis:notes "v README.md" Enter

tmux new-window -t diploma_thesis -n playground
tmux send-keys -t diploma_thesis:playground "activate diploma_thesis" Enter
tmux send-keys -t diploma_thesis:playground "cd playground" Enter
tmux send-keys -t diploma_thesis:playground "clear" Enter

tmux attach -t diploma_thesis:notes
