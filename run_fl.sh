#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python server.py --freeze 24 --img 1536 --adam --batch 1 --epochs 1 --data ./data/VisDrone.yaml --weights yolov5l.pt --hy data/hyps/hyp.VisDrone.yaml --cfg models/yolov5l-xs-tph.yaml --name fl_exp_2 &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 2); do
    echo "Starting client $i"
    python client.py --freeze 24 --img 1536 --adam --batch 1 --epochs 1 --data ./data/VisDrone.yaml --weights yolov5l.pt --hy data/hyps/hyp.VisDrone.yaml --cfg models/yolov5l-xs-tph.yaml --name fl_exp_2 --clientid "$i" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait