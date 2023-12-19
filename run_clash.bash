#!/bin/bash

CLASH_DIR="/hpcfs/fproject/linjw/Application/clash"
CLASH_PID_FILE="${CLASH_DIR}/clash.pid"

# 切换到clash目录
cd $CLASH_DIR

# 检查clash是否正在运行
if [ -f $CLASH_PID_FILE ]; then
    PID=$(cat $CLASH_PID_FILE)
    # 如果clash正在运行，则停止它
    if ps -p $PID > /dev/null; then
        echo "Stopping clash with PID $PID..."
        kill $PID
        rm $CLASH_PID_FILE
        git config --global --unset http.proxy
        git config --global --unset https.proxy
        unset http_proxy
        unset https_proxy
        echo "Clash stopped."
    else
        # PID文件存在，但进程不存在
        rm $CLASH_PID_FILE
        echo "Starting clash in background..."
        ./clash -d . & 
        echo $! > $CLASH_PID_FILE
        git config --global http.proxy http://127.0.0.1:7890
        git config --global https.proxy http://127.0.0.1:7890
        export http_proxy=http://127.0.0.1:7890
        export https_proxy=http://127.0.0.1:7890
        echo "Clash started with PID $!"
    fi
else
    # 如果clash没有运行，则启动它
    echo "Starting clash in background..."
    ./clash -d . & 
    echo $! > $CLASH_PID_FILE
    git config --global http.proxy http://127.0.0.1:7890
    git config --global https.proxy http://127.0.0.1:7890
    export http_proxy=http://127.0.0.1:7890
    export https_proxy=http://127.0.0.1:7890
    echo "Clash started with PID $!"
fi

