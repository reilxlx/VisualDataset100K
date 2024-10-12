## 本地部署大模型及构建VisualDataset100K数据集

使用vllm在本地部署大模型，并利用其构建VisualDataset100K数据集。

### 1. 本地部署大模型(vllm + nginx)

示例使用4块T4 GPU，通过vllm加载Qwen2-VL-2B-Instruct模型，并使用nginx进行负载均衡。

**1.1 启动vllm实例:**

每个GPU上运行一个vllm实例，端口分别为8001、8002、8003和8004。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8001 > backend1.log &

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8002 > backend2.log &

CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8003 > backend3.log &

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model /data/reilx/LLMode/qwen/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=4096 --port 8004 > backend4.log &
```

**1.2 配置nginx负载均衡:**

在nginx配置文件(`nginx.conf`)的`http`块中引入`vllm.conf`：

```nginx
http {
    include /usr/local/nginx/conf/vllm.conf;
    ...
}
```

`vllm.conf`内容如下：

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

配置完成后，重启nginx服务。


### 2. 构建VisualDataset100K数据集

利用已部署的模型，我们使用提供的Python脚本创建VisualDataset100K数据集。

**2.1 数据集生成脚本:**

* **`ImagesToQuestion_vllm_VD100K.py`**:  为每张图像生成问题，并将结果保存到JSON文件。
* **`ImagesToQuestionAns_vllm_VD100K.py`**:  基于生成的问题，生成对应答案。
* **`ImagesToDetails_vllm_VD100K.py`**:  生成图像的详细描述信息。


**2.2 VisualDataset100K数据集内容:**

本数据集包含以下几个部分：

* **图像详细描述数据集 (100K):**
    * `Qwen2VL2B_Details.jsonl`: 使用Qwen2VL-2B生成的图像描述。
    * `Qwen2VL7B_Details.jsonl`: 使用Qwen2VL-7B生成的图像描述。

* **图像问答对数据集 (100K & 58K):**
    * `QuestionsAnswers_Qwen2VL2B.jsonl`:  Qwen2VL-7B提问，Qwen2VL-2B回答 (100K)。
    * `QuestionsAnswers_Qwen2VL7B.jsonl`:  Qwen2VL-7B提问，Qwen2VL-7B回答 (100K)。
    * `QuestionsAnswers-Claude3_5sonnnet-sorted.jsonl`: Claude3.5Sonnet提问和回答 (58K)。
    * `QuestionsAnswers-Qwen2VL2B-sorted.jsonl`: Claude3.5Sonnet提问，Qwen2VL-2B回答 (58K)。
    * `QuestionsAnswers-Qwen2VL7B-sorted.jsonl`: Claude3.5Sonnet提问，Qwen2VL-7B回答 (58K)。
    * `QuestionsAnswers-Qwen2VL72B-sorted.jsonl`: Claude3.5Sonnet提问，Qwen2VL-72B回答 (58K)。

* **DPO数据集 (58K):** 用于Direct Preference Optimization训练。
    * `Claude-Qwen2VL2B.json`
    * `Claude-Qwen2VL7B.json`
    * `Qwen2VL72B-Qwen2VL2B.json`
    * `Qwen2VL72B-Qwen2VL7B.json`

* **SFT数据集 (58K):** 用于Supervised Fine-Tuning训练。
    * `QuestionsAnswers-Claude3_5sonnnet.json`
    * `QuestionsAnswers-Qwen2VL2B.json`
    * `QuestionsAnswers-Qwen2VL7B.json`
    * `QuestionsAnswers-Qwen2VL72B.json`
