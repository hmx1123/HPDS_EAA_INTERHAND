#!/bin/bash

echo "Start sequential model evaluation..."

names="hrnet_w32_YModel_vit512_gcn256_24_hms_only_all"
train="lr1en05_bs64_loss1p44"
#!/bin/bash

for i in {1..50}; do
    echo "========================================"
    echo "Start evaluating model: ${names}_${train}_${i}"
    echo "Start time: $(date)"
    
    # Define path variables
    encoder_path="./output/model/exp/${names}/${names}_${train}_encoder_${i}.pth"
    decoder_path="./output/model/exp/${names}/${names}_${train}_decoder_${i}.pth"
    config_path="./output/model/exp/${names}/${names}.yaml"
    output_file="./evals/eval_${names}_${train}_${i}.out"
    
    # Check if all required files exist
    missing_files=()
    
    if [[ ! -f "$encoder_path" ]]; then
        missing_files+=("Encoder: $encoder_path")
    fi
    
    if [[ ! -f "$decoder_path" ]]; then
        missing_files+=("Decoder: $decoder_path")
    fi
    
    if [[ ! -f "$config_path" ]]; then
        missing_files+=("Config file: $config_path")
    fi
    
    # Skip this model if any files are missing
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo "✗ Error: The following files are missing:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        echo "Skipping evaluation for model ${names}_${train}_${i}..."
        echo "========================================"
        echo
        continue
    fi
    
    echo "✓ Check passed: All required files exist"
    echo "  Encoder: $encoder_path"
    echo "  Decoder: $decoder_path"
    echo "  Config file: $config_path"
    echo "  Output file: $output_file"
    
    # Show file sizes
    echo "  File sizes:"
    echo "  - Encoder: $(du -h "$encoder_path" | cut -f1)"
    echo "  - Decoder: $(du -h "$decoder_path" | cut -f1)"
    
    # Create output directory if it doesn't exist
    output_dir="./evals"
    if [[ ! -d "$output_dir" ]]; then
        echo "Creating output directory: $output_dir"
        mkdir -p "$output_dir"
    fi
    
    # Execute the evaluation command
    echo "Starting evaluation command..."
    python apps/eval_interhand.py \
        --cfg "$config_path" \
        --encoder "$encoder_path" \
        --decoder "$decoder_path" \
        --data_path /home/hmx/hmx1123/datasets/interhand2.6m \
        > "$output_file" 2>&1

    # Check the exit status of the previous command
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "✓ Model ${names}_${train}_${i} evaluation completed successfully"
    else
        echo "✗ Model ${names}_${train}_${i} evaluation failed with exit code: $exit_status"
        
        # Show last few lines of error output
        echo "Last 10 lines of output:"
        tail -10 "$output_file"
    fi

    echo "End time: $(date)"
    echo "Output file: $output_file"
    echo "========================================"
    echo
done

echo "All model evaluations completed!"