# !/bin/bash

models=( 'gpt-4o' 'gemini-2.0-flash' 'claude-3-5-haiku-20241022' 'DeepSeek-V3')
tasks=( 'C' 'T' 'M')
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
      python eval-vignets.py --model_name "$model" --test_temperature 0 --GEMINI_API_KEY "<API_Key>" --Together_API_KEY "<API_Key>" --claude_API_KEY "<API_Key>" --Task "$task"
    done
done
