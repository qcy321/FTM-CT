#!/bin/bash
# 生成不重复的文件名
generate_new_filename() {
    local src_path="$1"
    local dest_dir="$2"
    local base_name=$(basename "$src_path")
    local dest_path="$dest_dir/$base_name"
    local counter=1

    # 如果目标路径不存在，直接返回原文件名
    if [[ ! -e "$dest_path" ]]; then
        echo "$dest_path"
        return 0
    fi

    # 分离文件名和扩展名
    local name="${base_name%.*}"
    local ext=""
    if [[ "$base_name" =~ \. ]]; then
        ext=".${base_name##*.}"
    fi

    # 递增计数，直到找到一个不存在的文件名
    while [[ -e "$dest_path" ]]; do
        dest_path="$dest_dir/${name}_${counter}${ext}"
        ((counter++))
    done

    echo "$dest_path"
    return 0
}

# 移动文件的函数，避免覆盖
move_files() {
    local model_dir="$1"
    local lang="$2"  # 第二个参数，用于替换 "python"
    local model_name="$3"
    local log="$4"
    local result="$5"
    local saved_models_dir="./saved_models/$lang/$model_name"

    echo "正在移动文件到目录：$model_dir"
    mkdir -p "$model_dir" || {
        echo "错误：无法创建目录 $model_dir"
        return 1
    }

    # 定义需要移动的文件和目录
    local items_to_move=(
        "$log"
        "$result"
        "$saved_models_dir"
    )

    # 移动文件并检查是否成功
    for item in "${items_to_move[@]}"; do
        if [[ -e "$item" ]]; then
            local dest_path=$(generate_new_filename "$item" "$model_dir")
            mv -v "$item" "$dest_path" || {
                echo "错误：无法移动 $item 到 $dest_path"
                return 1
            }
        else
            echo "警告：$item 不存在，跳过移动"
        fi
    done

    echo "文件移动完成。"
    return 0
}

# 用于存储所有通过脚本启动的进程 PID
declare -a PIDS=()

# 定义函数：在脚本终止时清理所有子进程
cleanup() {
    local exit_code=$?
    echo "脚本终止（退出码：$exit_code），正在清理所有子进程..."

    # 杀掉所有记录的 PID
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
            echo "终止进程 PID: $pid"
            kill -9 "$pid" 2>/dev/null || echo "警告：无法终止 PID $pid"
        fi
    done

    # 清空 PID 数组
    PIDS=()
    echo "所有子进程已终止"
    exit $exit_code
}

# 定义函数，等待某个程序执行完，再运行
wait_pid() {
	 local target_pid=$1

    echo "Waiting for process $target_pid to finish..."

    # 循环检查进程是否存在
    while kill -0 $target_pid 2>/dev/null; do
        sleep 60  # 每秒检查一次
    done

    echo "Process $target_pid has completed. Continuing..."
    # 后续操作
    echo "Next task is running now."
}

# 捕获多种信号和退出事件
trap cleanup INT TERM HUP EXIT

#    strategy的值选择
#    CROSS = "cross"
#    INCREASE = "increase"
#    DECREASE = "decrease"
#    DIVIDE = "divide"
#    NONE = "none"

run_model() {
    local model_name=$1       # 模型名称（如 codet5、plbart、roberta 等）
    local model_path=$2       # 模型路径（如 Salesforce/codet5-base 等）
    local lang=$3             # 语言（如 ruby 等）
    local queue_size=$4       # 队列大小（如 256、512、1024、2048 等）
    local dataset=$5          # 数据集名称（如 CSN 等）
    local hidden_method=$6    # 特征提取方式（如 avg 或 cls）
    local epoch=$7
    local strategy="cross"
    local checkpoint="best"
    #修改日志文件，适合启动多个任务时，进行更改，防止日志冲突。
    local log="train.log"
    local result="mrr_result.txt"
    local train_mode="finetune"

    # --- 新增：检查目标目录是否已存在 ---
    local target_dir="${dataset}/${lang}/${strategy}/${model_name}"
    if [[ -d "$target_dir" ]]; then
        echo "跳过任务 $model_name (${dataset}/${queue_size}/${model_name})，目标目录 $target_dir 已存在"
        return 0
    fi
    # --- 新增结束 ---

    echo "运行${model_name}模型(${dataset}/${queue_size}/${model_name})..."

    nohup python run.py \
        --do_train --do_test --fp16 \
        --lang "$lang" \
        --queue_size "$queue_size" \
        --model_name "$model_name" \
        --model_name_or_path="$model_path" \
        --hidden_state_method="$hidden_method" \
        --finetune_strategy="$strategy" \
        --finetune_checkpoint="$checkpoint" \
        --learning_rate 2e-5\
        --temperature 0.05\
        --device_ids 0 \
        --dataset "$dataset" \
        --train_batch_size 64 \
        --seed 42 \
        --log "$log" \
        --mrr_result "$result" \
        --train_mode "$train_mode" \
        --log_interval 100 \
        --num_train_epochs $epoch &

    local pid=$! # 获取当前任务的 PID
    echo "启动进程 PID: $pid"
    PIDS+=("$pid")  # 记录 PID 到数组中

    # 等待当前任务完成
    wait "$pid"
    local status=$?

    if [[ $status -eq 0 ]]; then
        echo "任务 $model_name (${dataset}/${queue_size}/${model_name}) 完成"
        move_files "${dataset}/${lang}/${queue_size}/${strategy}/${epoch}/${checkpoint}/${model_name}" "$lang" "$model_name" "$log" "$result"
    else
        echo "错误：任务 $model_name (${dataset}/${queue_size}/${model_name}) 失败，状态码 $status"
    fi

    # 从 PIDS 数组中移除已完成的 PID
    PIDS=("${PIDS[@]/$pid}")
    sleep 10
}

#wait_pid 3518002

# 声明关联数组
declare -A dataset_epochs

# 模型运行配置
dataset_epochs=(
    ["CSN"]=10
    #["AdvTest"]=4
    #["CosQA"]=10
    #["GPD"]=10
)

# 调试：打印关联数组内容
echo "关联数组内容："
declare -p dataset_epochs

# 模型运行配置
languages=("ruby" "javascript" "php" "java" "go" "python")
#languages=("python")

#队列大小
sizes=(2048)

# 遍历关联数组的键
for DATASET in "${!dataset_epochs[@]}"; do
    epoch=${dataset_epochs[$DATASET]}
    echo "DEBUG: dataset=$DATASET, epoch=$epoch"
    for la in "${languages[@]}"; do
        for size in "${sizes[@]}"; do
            #run_model "roberta" "roberta-base" "$la" $size "$DATASET" "avg" "$epoch"
            #run_model "codebert" "microsoft/codebert-base" "$la" $size "$DATASET" "avg" "$epoch"
            #run_model "unixcoder" "microsoft/unixcoder-base" "$la" $size "$DATASET" "avg" "$epoch"
            #run_model "cocosoda_time" "DeepSoftwareAnalytics/CoCoSoDa" "$la" $size "$DATASET" "avg" "$epoch"
            #run_model "codet5_time" "Salesforce/codet5-base" "$la" $size "$DATASET" "avg" "$epoch"
            #run_model "plbart_time" "uclanlp/plbart-base" "$la" $size "$DATASET" "avg" "$epoch"
            #run_model "graphcodebert" "microsoft/graphcodebert-base" "$la" $size "$DATASET" "cls" "$epoch"
            run_model "bge_code_time" "BAAI/bge-code-v1" "$la" $size "$DATASET" "avg" "$epoch"
            #run_model "qwen3_time" "Qwen/Qwen3-Embedding-0.6B" "$la" $size "$DATASET" "avg" "$epoch"
        done
    done
done

echo "所有任务已完成。"
