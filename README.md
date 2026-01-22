Official code for the paper ["Persona Jailbreaking in Large Language Models"]() accepted at EACL26 (Findings).

#### Requirements
Some of the important software dependecies are as follows:
* Python 3.9.21
* anthropic 0.49.0
* google-generativeai 0.8.3
* openai 1.60.2
* scikit-learn 1.6.1

We assume that you have installed conda beforehand. Use the following commands to replicate our environments using `openai.yml`. You can find the details of dependecies in these files. Make sure you export your openai key to the environment using `export OPENAI_API_KEY=YOUR_KEY`.

```
conda env create -f openai.yml
conda activate openai
cd codes
```

Execute all the codes from `./codes` path.


#### How to apply the PHISH framework?
Using the followng script, you can apply PHISH on any llm using guidelines. Make sure you have your API key for the respective llm provider. Refer to code for the respective flag to use specific llm provider.

```bash
python main.py --enable_attack 1 --baseline "$baseline" --dataset_name "$dataset" \
--model_name "$model"

```
Using the following flags, you can generate results for all the entries reported in the Table 1. To get more idea about other flags, refer to code. You may have to locally run medgemma-27b. Please refer to the code to get an idea. Note that the power of PHISH attack could be further increased by adding more number of target personality demonstrations.
```
model_name : gpt-4o o3-mini gemini-2.0-flash DeepSeek-V3 claude-3-5-haiku-20241022 meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 ChatHaruhi medgemma-27b 
baseline : BASE RANDOM SLIP  CipherCHAT DeepInception DrAttack FlipAttack PHISH100 PHISH150
dataset_name : BFI MPI ANTHR 
```
#### How to use CHATHARUHI baseline?
You need to create a new conda enviroment using `chatharushi.yml` file. We have used the [Chatharushi](https://github.com/LC1332/Chat-Haruhi-Suzumiya) library in our implementation. 

```bash
conda env create -f chatharushi.yml
conda activate chatharushi

```

#### How to reproduce the results in Table 1?
We have stored all assessment reports in Result section. The following script uses these stored reports to evaluate the results. Please note that if you choose to regenerate the reports using `main.py` you may not get the exact numbers reported in the Table 1 due to various factors such as stochastic nature of LLMs, exact version of the llm etc. However, you will find the similar trend of results reported in Table 1. 
```bash
python evaluation_script.py

```

#### How to produce analysis results?
* `ablation.sh` : Refer to this for ablation results.
* `trait_corr_analysis.py` : This reproduces the trait correlation results reported in Table 2.
* `eval_vignet.sh` :  Refer to this for evaluating PHISH on 3 high-risk scenarios reported in Section 5.4
* `llm-abilities-eval.py` : Refer to this for results of Section 5.5


```
#### How to cite our work?
```
TBD
```

#### License
This project is licensed under the terms of the `Apache license 2.0`.

#### Acknowledgements
This work was supported by the “R&D Hub Aimed at Ensuring Transparency and Reliability of Generative AI Models” project of the Ministry of Education, Culture, Sports, Science and Technology.


