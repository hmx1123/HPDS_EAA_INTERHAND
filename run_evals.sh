
#! /bin/bash

echo "Start sequential model evaluation..."

for i in {9.. 11}; do
    name="resnet34_hpds_eaa"
    echo "========================================"
    echo "Start evaluating model: hpds_eaa_distillation_${i}.pth"
    echo "Start time: $(date)"

    # Execute the evaluate command and wait for it to complete
    python apps/eval_interhand.py \
        --data_path /home/hmx/hmx1123/datasets/interhand2.6m \
        --model "./output/model/exp/${name}_${i}.pth" \
        > "./eval/eval_${i}.out" 2>&1

    # Check the exit status of the previous command
    if [ $? -eq 0 ];  then
        echo "✓ Model ${name}_${i}.pth evaluation completed successfully"
    else
        echo "✗ model ${name}_${i}.pth evaluation failed"
    fi

    echo "End time: $(date)"
    echo "Output file: eval_${i}.out"
    echo "========================================"
    echo
done

echo "All model evaluations completed!"