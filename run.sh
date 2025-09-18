#!/bin/bash
# Generate a unique filename
generate_new_filename() {
    local src_path="$1"
    local dest_dir="$2"
    local base_name=$(basename "$src_path")
    local dest_path="$dest_dir/$base_name"
    local counter=1

    # If the destination path does not exist, return the original filename
    if [[ ! -e "$dest_path" ]]; then
        echo "$dest_path"
        return 0
    fi

    # Separate filename and extension
    local name="${base_name%.*}"
    local ext=""
    if [[ "$base_name" =~ \. ]]; then
        ext=".${base_name##*.}"
    fi

    # Increment counter until a non-existent filename is found
    while [[ -e "$dest_path" ]]; do
        dest_path="$dest_dir/${name}_${counter}${ext}"
        ((counter++))
    done

    echo "$dest_path"
    return 0
}

# Move files function to avoid overwriting
move_files() {
    local model_dir="$1"
    local lang="$2"  # Second parameter, used to replace "python"
    local model_name="$3"
    local log="$4"
    local result="$5"
    local saved_models_dir="./saved_models/$lang/$model_name"

    echo "Moving files to directory: $model_dir"
    mkdir -p "$model_dir" || {
        echo "Error: Failed to create directory $model_dir"
        return 1
    }

    # Define items to move
    local items_to_move=(
        "$log"
        "$result"
        "$saved_models_dir"
    )

    # Move files and check if successful
    for item in "${items_to_move[@]}"; do
        if [[ -e "$item" ]]; then
            local dest_path=$(generate_new_filename "$item" "$model_dir")
            mv -v "$item" "$dest_path" || {
                echo "Error: Failed to move $item to $dest_path"
                return 1
            }
        else
            echo "Warning: $item does not exist, skipping move"
        fi
    done

    echo "File move completed."
    return 0
}

#Used to store all process PIDs started by the script
declare -a PIDS=()

# Define function: clean up all subprocesses when the script terminates
cleanup() {
    local exit_code=$?
    echo "Script terminated (exit code: $exit_code), cleaning up all child processes..."

    # Kill all recorded PIDs
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
            echo "Terminating process PID: $pid"
            kill -9 "$pid" 2>/dev/null || echo "Warning: Failed to terminate PID $pid"
        fi
    done

    # Clear the PID array
    PIDS=()
    echo "All child processes have terminated"
    exit $exit_code
}

# Define a function to wait for a program to finish executing and then run
wait_pid() {
	 local target_pid=$1

    echo "Waiting for process $target_pid to finish..."

    # Loop to check if the process exists
    while kill -0 $target_pid 2>/dev/null; do
        sleep 60  # Check once per second
    done

    echo "Process $target_pid has completed. Continuing..."
    # Subsequent Operations
    echo "Next task is running now."
}

# Capture various signals and exit events
trap cleanup INT TERM HUP EXIT

# strategy value selection
#    CROSS = "cross"
#    INCREASE = "increase"
#    DECREASE = "decrease"
#    DIVIDE = "divide"
#    NONE = "none"

run_model() {
    local model_name=$1        # Model name (e.g., codet5, plbart, roberta)
    local model_path=$2        # Model path (e.g., Salesforce/codet5-base)
    local lang=$3              # Language (e.g., Ruby)
    local queue_size=$4        # Queue size (e.g., 256, 512, 1024, 2048)
    local dataset=$5           # Dataset name (e.g., CSN)
    local hidden_method=$6     # Feature extraction method (e.g., avg or cls)
    local epoch=$7
    local strategy="cross"
    local checkpoint="best"
    #Modify the log file. It is suitable for starting multiple tasks and making changes to prevent log conflicts.
    local log="train.log"
    local result="mrr_result.txt"
    local train_mode="finetune"

    # Define target directory
    local target_dir="${dataset}/${lang}/${strategy}/${model_name}"
    if [[ -d "$target_dir" ]]; then
        echo "Skipping task $model_name (${dataset}/${queue_size}/${model_name})，Target directory $target_dir already exists"
        return 0
    fi

    echo "running ${model_name} model (${dataset}/${queue_size}/${model_name})..."

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

    local pid=$! # Capture the PID of the background process
    echo "Start a process PID: $pid"
    PIDS+=("$pid")  # Add PID to the array

    # Wait for the current task to complete
    wait "$pid"
    local status=$?

    if [[ $status -eq 0 ]]; then
        echo "Task $model_name (${dataset}/${queue_size}/${model_name}) completed successfully"
        move_files "${dataset}/${lang}/${queue_size}/${strategy}/${epoch}/${checkpoint}/${model_name}" "$lang" "$model_name" "$log" "$result"
    else
        echo "Error：task $model_name (${dataset}/${queue_size}/${model_name}) failed with status code $status"
    fi

    # Remove the completed PID from the PIDS array
    PIDS=("${PIDS[@]/$pid}")
    sleep 10
}

# Wait for a process to complete before running
#wait_pid 3518002

# Declare an associative array
declare -A dataset_epochs

# Select the dataset and epoch size to run
dataset_epochs=(
    ["CSN"]=10
    #["AdvTest"]=4
    #["CosQA"]=10
    #["GPD"]=10
)

# Debug: Print associative array contents
echo "Associative array contents:"
declare -p dataset_epochs

# Model run configuration
languages=("ruby" "javascript" "php" "java" "go" "python")
#languages=("python")

#queue size
sizes=(2048)

# Iterate over the keys of an associative array
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

echo "All tasks have been completed."
