#!/bin/zsh

autoload -Uz tmux_create_or_attach_session
tmux_create_or_attach_session "diploma_thesis" "notes"

if [ $? -eq 1 ]; then
  tmux send-keys -t diploma_thesis:notes "v README.md" Enter

  tmux new-window -t diploma_thesis -n playground
  tmux send-keys -t diploma_thesis:playground "activate diploma_thesis" Enter
  tmux send-keys -t diploma_thesis:playground "cd playground" Enter
  tmux send-keys -t diploma_thesis:playground "clear" Enter

  tmux attach -t diploma_thesis:notes
fi

