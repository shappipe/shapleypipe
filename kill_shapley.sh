#!/bin/sh

if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "Error: match processes with a keyword" >&2
    echo "usage: $0 <keyword>" >&2
    exit 1
fi

keyword="$1"

pids=$(ps aux | grep -F "$keyword" | grep -v -F 'grep' | grep -v -F "$0" | awk '{print $2}' | tr '\n' ' ')

pids=$(echo "$pids" | sed 's/^ *//;s/ *$//')

if [ -z "$pids" ]; then
    echo "cannot find processes including '$keyword'"
    exit 0
fi

echo "found processes with keyword '$keyword':"
ps aux | grep -F "$keyword" | grep -v -F 'grep' | grep -v -F "$0"

printf "\nAre you sure to kill the processes? (y/N): "
read answer

case $answer in
    [yY]|[yY][eE][sS])
        # kill
        echo "Terminating processes..."
        echo "$pids" | xargs kill -TERM
        echo "Sent SIGTERM to process: $pids"
        ;;
    *)
        echo "Canceled"
        exit 0
        ;;
esac