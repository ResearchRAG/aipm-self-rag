# 自我反思检索增强生成模型（SELF-RAG）：通过自我反思学习检索、生成和批评

这包括了[自我反思检索增强生成模型（SELF-RAG）：通过自我反思学习检索、生成和批评](https://arxiv.org/abs/2310.11511)（ICLR 2024，口头报告前1%）的原始实现，作者为Akari Asai、Zeqiu Wu、Yizhong Wang、Avirup Sil和Hannaneh Hajishirzi。

[网站](https://selfrag.github.io/) | [7B模型](https://huggingface.co/selfrag/selfrag_llama2_7b) | [13B模型](https://huggingface.co/selfrag/selfrag_llama2_13b) | [论文](https://akariasai.github.io/files/adaptive_retrieval_augmented_lm_arxiv.pdf) | [训练数据](https://huggingface.co/datasets/selfrag/selfrag_train_data) | [推特摘要](https://twitter.com/AkariAsai/status/1715110277077962937) | [更新](#更新)

**自我反思检索增强生成模型（Self-RAG）**（图右）是一个新框架，用于训练任意大型语言模型（LM）学习检索、生成和批评，以提高生成事实性和质量，同时不损害大型语言模型（LLM）的多功能性。

与传统的检索增强生成（RAG；图左）方法不同，**自我反思检索增强生成模型**根据多样化的查询按需检索（例如，可以多次检索或完全跳过检索），并通过预测**反思标记**从多个细粒度方面批评自己的生成，将反思标记作为生成的一个组成部分。

我们进行逐段的束搜索，以选择最大化多样化偏好的效用的输出。

![](images/teaser_self_rag_v8.png)

如果你发现我们的代码、数据、模型或论文有用，请引用论文：
```
@inproceedings{
asai2024selfrag,
author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
title={Self-{RAG}: Learning to Retrieve, Generate, and Critique through Self-Reflection},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=hSyW5go0v8} 
}
```

## 更新
- **2023.10**：代码、模型和论文的初始发布。

## 内容
1. [安装](#安装)
2. [快速开始](#快速开始)
2. [检索器设置](#检索器设置)
3. [训练](#训练)
4. [推理](#推理)
5. [基线](#基线)
6. [常见问题解答](#常见问题解答)
7. [联系方式](#联系方式)

## 安装
通过运行下面的命令安装依赖的Python库。

```
pip install -r requirements.txt
```
请使用最新版本的`vllm`，因为旧版本可能无法通过`SamplingParam`设置`skip_special_tokens`，这是通过（[this PR](https://github.com/vllm-project/vllm/issues/893)）添加的。

你也可以通过运行下面的命令创建一个conda环境。

```
conda env create -f environment.yml
```

## 快速开始
你可以从HuggingFace Hub下载Self-RAG。对于推理，我们建议使用[vllm](https://vllm.readthedocs.io/en/latest/)，因为它显著加快了推理速度。

```py
from vllm import LLM, SamplingParams

model = LLM("selfrag/selfrag_llama2_7b", download_dir="/gscratch/h2lab/akari/model_cache", dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt

query_1 = "Leave odd one out: twitter, instagram, whatsapp."
query_2 = "Can you tell me the difference between llamas and alpacas?"
queries = [query_1, query_2]

# 对于不需要检索的查询
preds = model.generate([format_prompt(query) for query in queries], sampling_params)
for pred in preds:
  print("Model prediction: {0}".format(pred.outputs[0].text))
```

输出：
```txt
Model prediction: Twitter, Instagram, and WhatsApp are all social media platforms. [No Retrieval]WhatsApp is the odd one out because it is a messaging app, while Twitter and # Instagram are primarily used for sharing photos and videos.[Utility:5]</s>
Model prediction: Sure![Retrieval]<paragraph><paragraph>
```
如你所见，当第一个查询不需要检索时，Self-RAG开始生成不包含检索的响应。另一方面，对于第二个查询，Self-RAG输出了`[Retrieve]`标记，因为这个问题需要更细粒度的事实基础。

对于需要事实基础的查询，你可以插入一个段落。Self-RAG可以在生成过程中随时检索和插入段落，并在它们被上下文标记特殊标记`<paragraph>`和`</paragraph>`包围时识别它们。
```
# 对于需要事实基础的查询
prompt = format_prompt("Can you tell me the difference between llamas and alpacas?", "The alpaca (Lama pacos) is a species of South American camelid mammal. It is similar to, and often confused with, the llama. Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber.")
preds = model.generate([prompt], sampling_params)
print([pred.outputs[0].text for pred in preds])
# ['[Relevant]Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber.[Fully supported][Utility:5]</s>']
```
Self-RAG找到了相关插入的文档，并生成了完全由证据支持的答案。

**请注意，这个演示使用了较小的语料库和具有完整推理算法的Self-RAG。对于完整评估，你需要设置检索器或下载我们检索的结果。请按照[推理](#instruction)中的说明操作。**

## 检索器设置
默认情况下，我们使用[Contriever](https://github.com/facebookresearch/contriever)作为我们的检索组件。

### 下载数据
下载DPR使用的预处理段落数据。
```
cd retrieval_lm
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz 
```

然后，下载生成的段落。我们使用[Contriever-MSMARCO](https://huggingface.co/facebook/contriever-msmarco) 
```
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar 
```

### 运行检索器
你可以通过运行下面的命令来运行段落检索。

```
cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data YOUR_INPUT_FILE  \
    --output_dir YOUR_OUTPUT_FILE \
    --n_docs 20
```
你的输入文件应该是一个`json`或`jsonl`。每个实例必须包含`question`或`instruction`，这些将在检索期间作为查询使用。

### 为你自己的数据生成嵌入

你可以通过运行以下命令为你自己的数据生成嵌入。（该脚本改编自Contriever仓库。）请注意，从大规模语料库（>10M文档）生成嵌入可能需要时间，我们建议在多个GPU上运行。

```
cd retrieval_lm
for i in {0..3}; do
  export CUDA_VISIBLE_DEVICES=${i}
  python generate_passage_embeddings.py  --model_name_or_path facebook/contriever-msmarco \
  --output_dir YOUR_OUTPUT_DIR \
  --passages YOUR_PASSAGE_DATA --shard_id ${i}  --num_shards 4 > ./log/nohup.my_embeddings.${i} 2>&1 &
```

## 训练
**自我反思检索增强生成模型**训练两个模型，*Critic*和*Generator*，这两个模型都通过标准下一个标记预测目标扩展标记词汇表。

- [步骤1：收集反思标记](#collect-reflection-tokens)：使用GPT4生成Critic训练数据。
- [步骤2：Critic训练](#critic-training)：使用新的特殊标记训练Critic。
- [步骤3：生成器数据创建](#generator-data-creation)：使用Critic和检索器生成生成器训练数据。
- [步骤4：生成器训练](#generator-training)：使用新的特殊标记训练生成器。

或者，你可以下载我们包含150K实例的训练数据[这里](https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view?usp=share_link)。

### 收集反思标记
我们从GPT-4收集训练数据。调用GPT-4的脚本每种特殊标记类型都在[data_creation/critic](data_creation/critic)。

或者，你可以在[这里](https://drive.google.com/file/d/1IN1XcIOYtRIGWITJ4LKRgfITT-uUwk_W/view?usp=share_link)下载我们的训练数据。

### Critic训练
一旦你创建或下载了训练数据，运行下面的命令对Llama2-7B进行Critic训练。
```
cd data_creation
torchrun --nproc_per_node=2 \
  --master_port=2568 train_special_tokens.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path PATH_TO_TRAIN_DATA_FILE \
  --bf16  True \
  --output_dir PATH_TO_CRITIC_MODEL \
  --num_train_epochs 3  \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --fsdp "full_shard auto_wrap"
```

### 生成器数据创建
生成器训练数据的代码在[data_creation/generator](data_creation/generator)下。请参阅[README.md](data_creation/generator/README.md)中的说明。

或者，你可以在[HuggingFace dataset](https://huggingface.co/datasets/selfrag/selfrag_train_data/tree/main)或[这里](https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view?usp=share_link)下载我们的训练数据

### 生成器训练
对于生成器训练，我们使用DeepSpeed使训练更高效。设置训练数据路径后，通过运行下面的脚本来运行训练。

```
cd retrieval_lm
bash script_finetune_7b.sh
```
对于13B模型训练，使用`training_13b`。我们使用8 A100与40 GRAM进行7B模型训练，使用4 a100与80 GB GRAM进行13B训练。7B应该适合1-2 A100，尽管训练可能会很慢。

## 推理
对于论文中进行的任务评估，请下载数据[这里](https://drive.google.com/file/d/1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb/view?usp=share_link)。

每个文件已经附带了由Contriever离线检索的文档，所以如果你不想在推理过程中运行检索器，你可以直接在`contexts`加载检索的文档。

下面，我们描述了Self-RAG和基线。
- [短文本](#shot_form)：运行短文本生成的评估。
- [长文本](#long_form)：运行长文本生成的评估。

### 短文本（PubHealth、ARC-Challenge、TriviaQA、PopQA）
由于我们通常只对短文本生成任务检索一次，我们提供了一个易于运行的评估脚本，利用Contriever预先检索的文档。请参阅下面的个别命令。

#### 问答

```
python run_short_form.py \
--model_name selfrag/selfrag_llama2_7b \
--input_file eval_data/popqa_longtail_w_gs.jsonl \
--mode MODE --max_new_tokens 100 \
--threshold 0.2 \
--output_file YOUR_OUTPUT_FILE \
--metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
--dtype half
```

`mode`指定了推理时间模式中的`['adaptive_retrieval', 'no_retrieval', 'always_retrieve']`。

- `adaptive_retrieval`根据`threshold`或Self-RAG预测进行检索
- `no_retrieval`在推理时禁用检索
- `always_retrieve`总是检索。

对于13B，如果你使用单个GPU与24 GRAM，可能会有OOM问题。你可以通过设置`--world_size`在多个GPU上运行推理。

#### ARC Challenge
```
python run_short_form.py \
  --model_name selfrag/selfrag_llama2_7b \
  --input_file eval_data/arc_challenge_processed.jsonl \
  --max_new_tokens 50 --threshold 0.2 \
  --output_file OUTPUT_FILE_NAME \
  --metric match --ndocs 5 --use_groundness --use_utility --use_seqscore \
  --task arc_c
```

#### PubHealth
```
python run_short_form.py \
  --model_name selfrag/selfrag_llama2_7b \
  --input_file eval_data/health_claims_processed.jsonl \
  --max_new_tokens 50 \
  --threshold 0.2 --output_file OUTPUT_FILE_NAME \
  --metric match --ndocs 5 \
  --use_groundness --use_utility --use_seqscore \
  --task fever
```

### 长文本（ASQA、FactScore）
对于长文本QA，你可以使用检索模型进行评估，也可以使用预先给定的段落进行评估。
目前，我们正在努力减少运行时内存需求（DPR / Contriever使用整个英文维基百科嵌入需要100 GB RAM）加速长文本生成，并发布使用小批量初始检索文档的推理代码（~20）。

*请注意：我们当前的实现专门针对目标任务数据集的评估。我们计划更新我们的代码库，使界面更简单、更易于使用。我们会在发布另一个版本时宣布。*

#### 使用预先检索的段落运行推理

对于ASQA，请运行以下命令，
```
python run_long_form_static.py \
  --model_name selfrag/selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --task asqa --input_file eval_data/asqa_eval_gtr_top100.json \
  --output_file YOUR_OUTPUT_FILE_NAME --max_depth 7 --mode always_retrieve \
```

对于FactScore，

```
python run_long_form_static.py \
  --model_name selfrag/selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --task factscore --input_file eval_data/factscore_unlabeled_alpaca_13b_retrieval.jsonl \
  --output_file YOUR_OUTPUT_FILE_NAME --max_depth 7 \
```

#### 长文本生成的关键参数
有几个关键参数与Self-RAG的推理有关。
- `w_rel`（默认1.0）：`w_rel`在束搜索中控制对`isRel`（关于检索段落是否相关的批评标记）标记概率的强调。
- `w_sup`（默认1.0）：`w_sup`在束搜索中控制对`isSup`（关于生成是否由文档支持的批评标记）标记概率的强调。
- `w_use`（默认0.5）：`w_use`在束搜索中控制对`isUse`（关于整体质量的批评标记）标记概率的强调。
- `threshold`（默认0.2）：这个阈值控制自适应检索的频率。
- `max_depth`（默认6）：这对应于论文中的`T`，它定义了搜索的最大深度。
- `beam_width`（默认2）：这控制段级束搜索中的束大小。

有关更多详细信息，请参阅我们论文中的详细说明（第3节）和分析（第5节）。

#### 运行评估
对于长文本评估，设置外部库或存储库以运行评估。

- `factscore==v0.1.5`（生物）
请按照[FactScore](https://github.com/shmsw25/FActScore)官方存储库的说明设置你的环境。
```
python -m factscore.factscorer --data_path YOUR_OUTPUT_FILE  --model_name retrieval+ChatGPT --cache_dir YOUR_CACHE_DIR --openai_key YOUR_OPEN_AI_KEY --verbose
```

- [ALCE/ASQA](https://github.com/princeton-nlp/ALCE) 

ALCE为长文本QA提供了使用多种不同指标的综合评估。对于你的第一次评估，安装ALCE存储库并下载数据。
```
git clone https://github.com/princeton-nlp/ALCE.git 
python3 -m alce_env
cd ALCE
bash download_data.sh
```

对于ASQA，你可以按如下方式运行评估。请注意，ASQA评估需要T5-XXL（11B）基于NLI模块。
```
python eval.py --f YOUR_OUTPUT_FILE --citations --qa --mauve
```

## 基线
重新运行基线的代码可在[run_baseline_lm.py](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_baseline_lm.py)。 
要运行检索增强基线，请确保下载带有检索段落的任务输入文件。

### 纯LM基线

- Huggingface模型
```
python run_baseline_lm.py \
--model_name meta-llama/Llama-2-7b-hf \
--input_file INPUT_FILE_SAME_AS_SELF_RAG \
 --max_new_tokens 100 --metric match \
--result_fp RESULT_FILE_PATH --task qa --prompt_name "prompt_no_input"
```
例如，PubHealth
```
python run_baseline_lm.py \
--model_name meta-llama/Llama-2-7b-hf \
--input_file eval_data/health_claims_processed.jsonl \
--max_new_tokens 20 \
--metric accuracy \
--result_fp llama2_7b_pubhealth_results.json \
--task fever
```
**注意：对于PubHealth和ARC，请传递任务名称（ARC = `arc_c` 和 PubHealth = `fever`）以正确设置指令。**
- OpenAI API

对于OpenAI API模型，你还需要在这里设置组织密钥[这里](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_baseline_lm.py#L12)。 你还需要有一个包含你的OpenAI API密钥的txt文件。
```
python run_baseline_lm.py \
--model_name gpt-3.5-turbo-0301 \
--input_file INPUT_FILE_SAME_AS_SELF_RAG \
--max_new_tokens 100 --metric match \
--result_fp RESULT_FILE_PATH \
 --task qa \
--api_key YOUR_OPEN_AI_API_KEY_FILE \
--prompt_name "prompt_no_input"
```

### 检索增强基线

- Huggingface模型

```
python run_baseline_refactor.py \
--model_name meta-llama/Llama-2-7b-hf \
--input_file INPUT_FILE_SAME_AS_SELF_RAG \
 --max_new_tokens 100 --metric match \
--result_fp RESULT_FILE_PATH --task qa \
--mode retrieval \
--prompt_name "prompt_no_input_retrieval"
```

- OpenAI API
```
python run_baseline_lm.py \
--model_name gpt-3.5-turbo-0301 \
--input_file INPUT_FILE_SAME_AS_SELF_RAG \
--max_new_tokens 100 --metric match \
--result_fp RESULT_FILE_PATH \
 --task qa \
--api_key YOUR_OPEN_AI_API_KEY_FILE \
--mode retrieval \
--prompt_name "prompt_no_input_retrieval"
```

## 常见问题解答
**Q1: 我如何使用Self-RAG方案训练一个新的预训练LM？** -- 如果你使用hugging face transformers，你可以简单地在训练脚本中更改`model_name_or_path`和`tokenizer_name`，[script_finetune_7b.sh](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/script_finetune_7b.sh)。如果你想使用你自己的微调脚本，请确保添加特殊标记并遮蔽段落上下文，如[这个问题](https://github.com/AkariAsai/self-rag/issues/12)中讨论的

**Q2: 你们计划发布基于Mirstral-7B的Self-RAG吗？** -- 现在，我有限的带宽无法做到这一点，但有一个社区训练版本的Self-RAG [SciPhi-Self-RAG-Mistral-7B-32k](https://huggingface.co/SciPhi/SciPhi-Self-RAG-Mistral-7B-32k) 在Mistral-7B之上。如果我们能在Mistral-7B上训练Self-RAG并发布检查点，我们会宣布。

## 联系方式
如果你有问题，请提出一个提及@AkariAsai的问题，或发送电子邮件至akari[at]cs.washington.edu。

