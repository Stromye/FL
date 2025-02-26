#!/bin/bash

 module load miniforge/24.9.2
# conda list
# which conda        

source activate torch  
# 禁用代理
unset http_proxy
unset https_proxy
# 启动服务端（在后台运行）
python server.py &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# 等待服务端启动
sleep 5  # 等待 5 秒
# 循环启动 4 个客户端，分别使用不同的 GPU
for ((i = 0; i < 4; i++)); do
    CUDA_VISIBLE_DEVICES=$i python client.py &
    CLIENT_PID[$i]=$!
    echo "Client $i started with PID: ${CLIENT_PID[$i]}"
done

# 等待所有进程结束
wait $SERVER_PID ${CLIENT_PID[@]}
