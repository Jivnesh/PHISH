# !/bin/bash

model='gpt-4o'
baselines=('setting0' 'setting1' 'setting2' 'setting3' 'setting4' 'setting5' )
# Baselines: 'UAS' 'SLIP' 'DAN' 'PHISH' 'RANDOM' 'BASE'
datasets=('BFI')

for dataset in "${datasets[@]}"; do
    for baseline in "${baselines[@]}"; do
      python ablation.py --enable_attack 1 --baseline "$baseline" --dataset_name "$dataset" \
      --model_name "$model"
    done
done

model='gemini-2.0-flash'
baselines=('setting0' 'setting1' 'setting2' 'setting3' 'setting4' 'setting5' )
# Baselines: 'UAS' 'SLIP' 'DAN' 'PHISH' 'RANDOM' 'BASE'
datasets=('BFI')

for dataset in "${datasets[@]}"; do
    for baseline in "${baselines[@]}"; do
      python ablation.py --enable_attack 1 --baseline "$baseline" --dataset_name "$dataset" \
      --model_name "$model" \
      --GEMINI_API_KEY "<API_KEY>" 
    done
done

model='claude-3-5-haiku-20241022'
baselines=('setting0' 'setting1' 'setting2' 'setting3' 'setting4' 'setting5' )
# Baselines: 'UAS' 'SLIP' 'DAN' 'PHISH' 'RANDOM' 'BASE'
datasets=('BFI')

for dataset in "${datasets[@]}"; do
    for baseline in "${baselines[@]}"; do
      python ablation.py --enable_attack 1 --baseline "$baseline" --dataset_name "$dataset" \
      --model_name "$model" \
      --claude_API_KEY "<API_KEY>"
    done
done

model='DeepSeek-V3'
baselines=('setting0' 'setting1' 'setting2' 'setting3' 'setting4' 'setting5' )
# Baselines: 'UAS' 'SLIP' 'DAN' 'PHISH' 'RANDOM' 'BASE'
datasets=('BFI')

for dataset in "${datasets[@]}"; do
    for baseline in "${baselines[@]}"; do
      python ablation.py --enable_attack 1 --baseline "$baseline" --dataset_name "$dataset" \
      --model_name "$model" \
      --Together_API_KEY "<API_KEY>" 
    done
done