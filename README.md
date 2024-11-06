[中文](README_zh.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           [Huggingface-VisualDataset100K](https://huggingface.co/datasets/REILX/VisualDataset100K)
## Local Deployment of Large Models and Construction of VisualDataset100K Dataset

Deploy large models locally using vllm and utilize them to construct the VisualDataset100K dataset.

### 1. Local Deployment of Large Models (vllm + nginx)

Uses multi GPUs, loads the Qwen/Qwen2-VL-2B-Instruct、Qwen/Qwen2-VL-7B-Instruct、Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4 models through vllm, and uses nginx for load balancing.

**1.1 Launch vllm instances:**

Run a vllm instance on each GPU, with ports 8001, 8002, 8003, and 8004 respectively.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8001 > backend1.log &

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8002 > backend2.log &

CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8003 > backend3.log &

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8004 > backend4.log &
```

**1.2 Configure nginx load balancing:**

Include `vllm.conf` in the `http` block of the nginx configuration file (`nginx.conf`):

```nginx
http {
    include /usr/local/nginx/conf/vllm.conf;
    ...
}
```

The content of `vllm.conf` is as follows:

```nginx
upstream vllm_backends {
    server 127.0.0.1:8001 weight=1;
    server 127.0.0.1:8002 weight=1;
    server 127.0.0.1:8003 weight=1;
    server 127.0.0.1:8004 weight=1;
}

server {
    listen 8000;

    location /v1/chat/completions {
        proxy_pass http://vllm_backends;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

After configuration, restart the nginx service.

### 2. Building VisualDataset100K Dataset

Using the deployed model, we create the VisualDataset100K dataset using the provided Python scripts.

**2.1 Dataset Generation Scripts:**

* **`ImagesToQuestion_vllm_VD100K.py`**: Generates questions for each image and saves results to JSON files.
* **`ImagesToQuestionAns_vllm_VD100K.py`**: Generates corresponding answers based on generated questions.
* **`ImagesToDetails_vllm_VD100K.py`**: Generates detailed descriptions of images.
* **`ImagesToChoiceQA_vllm_VD100K.py`**: Generates multiple-choice questions and answers for each image.
* **`JsonlChoiceQAClean.py`**: Organizes the json generated by ImagesToChoiceQA_vllm_VD100K.py.

**2.2 VisualDataset100K Dataset Contents:**

This dataset includes the following parts:

* **Detailed Image Description Dataset (100K):**
    * `Qwen2VL2B_Details.jsonl`: Image descriptions generated using Qwen2VL-2B.
    * `Qwen2VL7B_Details.jsonl`: Image descriptions generated using Qwen2VL-7B.
    * `Qwen2VL72BInt4_Details.jsonl`: Image descriptions generated using Qwen2VL-72B-Int4.

* **Image Q&A Dataset (100K & 58K):**
    * `Questions_Qwen2VL7B.jsonl`: Questions generated by Qwen2VL-7B based on image content (100K).
    * `QuestionsAnswers_Qwen2VL2B.jsonl`: Questions by Qwen2VL-7B, answers by Qwen2VL-2B (100K).
    * `QuestionsAnswers_Qwen2VL7B.jsonl`: Questions by Qwen2VL-7B, answers by Qwen2VL-7B (100K).
    * `QuestionsAnswers_Qwen2VL72BInt4.jsonl`: Questions by Qwen2VL-7B, answers by Qwen2VL-72B-Int4 (100K).
    * `QuestionsAnswers-Claude3_5sonnnet-sorted.jsonl`: Questions and answers by Claude3.5Sonnet (58K).
    * `QuestionsAnswers-Qwen2VL2B-sorted.jsonl`: Questions by Claude3.5Sonnet, answers by Qwen2VL-2B (58K).
    * `QuestionsAnswers-Qwen2VL7B-sorted.jsonl`: Questions by Claude3.5Sonnet, answers by Qwen2VL-7B (58K).
    * `QuestionsAnswers-Qwen2VL72B-sorted.jsonl`: Questions by Claude3.5Sonnet, answers by Qwen2VL-72B (58K).

* **Image-Based Multiple Choice Questions (100K):**
    * `Qwen2VL7B_ChoiceQA.jsonl`: Questions, four options, and answers generated by Qwen2VL-7B based on images (100K).
    * `Qwen2VL72BInt4_ChoiceQA.jsonl`: Questions, four options, and answers generated by Qwen2VL-72B-Int4 based on images (100K).

* **DPO Dataset (58K):** For Direct Preference Optimization training.
    * `Claude-Qwen2VL2B.json`
    * `Claude-Qwen2VL7B.json`
    * `Qwen2VL72B-Qwen2VL2B.json`
    * `Qwen2VL72B-Qwen2VL7B.json`

* **SFT Dataset (58K):** For Supervised Fine-Tuning training.
    * `QuestionsAnswers-Claude3_5sonnnet.json`
    * `QuestionsAnswers-Qwen2VL2B.json`
    * `QuestionsAnswers-Qwen2VL7B.json`
    * `QuestionsAnswers-Qwen2VL72B.json`

### 3. Download
The above dataset can be downloaded through Huggingface, [VisualDataset100K](https://huggingface.co/datasets/REILX/VisualDataset100K)

### Acknowledgments

This project benefits from the [Visual Genome Dataset V1.2](http://visualgenome.org/api/v0/api_home.html). Thanks to all the authors mentioned above for their contributions.

### If you found this project helpful, please give me a Star ⭐.
