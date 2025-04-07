<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
 *
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 *
http://www.apache.org/licenses/LICENSE-2.0
 *
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Quick Start

## Installing from source

### Prerequisites

Before you begin using Context Aware RAG, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

1. Clone the repository

```bash
git clone https://github.com/NVIDIA/context-aware-rag.git
cd context-aware-rag
```

2. Create a Python environment

```bash
uv venv --seed .venv
source .venv/bin/activate
```

3. Install the Context Aware RAG library

```bash
uv pip install -e .
```

4. Optionally: Build the wheel file
```bash
uv build
```

5. Install the wheel file
```bash
uv pip install dist/vss-ctx-rag-0.5.0-py3-none-any.whl
```


## Running depedent containers and setting up environment variables

Depending on the type of RAG you are using, you will need to run the following containers.

#### Graph-RAG: neo4j

``` bash
docker run -d \
  --name neo4j \
  -p <NEO4J_PORT>:7687 \
  -e NEO4J_AUTH=<GRAPH_DB_USERNAME>/<GRAPH_DB_PASSWORD> \
  neo4j:5.26.4
```

#### Vector-RAG: milvus

``` bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh


bash standalone_embed.sh start
```

This will start the milvus service by default on port 19530.


#### ENV Setup

#### Getting NVIDIA API Key
NVIDIA_API_KEY is NVIDIA Personal Key to use LLM and Rerank and Embeddings NIMs from build.nvidia.com. This key is essential for accessing NVIDIA’s cloud services and models. Here are the steps to get the NVIDIA API Key:

1. Log in to https://build.nvidia.com/explore/discover.

2. Navigate to any NIM e.g. https://build.nvidia.com/meta/llama3-70b.

3. Search for Get API Key on the page and click on it.

![Get API Key](../_static/getAPIkey.png)

4. Click on Generate Key.

![Generate Key](../_static/buttontogenerateAPIkey.png)

5. Store the generated API Key securely for future use.

Now setup the environment variables depending on the type of RAG.

#### Vector-RAG

``` bash
export MILVUS_HOST=<MILVUS_HOST_IP>
export MILVUS_PORT=<MILVUS_DB_PORT>
export NVIDIA_API_KEY=<NVIDIA_API_KEY>
```

#### Graph-RAG

``` bash
export GRAPH_DB_URI=bolt://<HOST>:<NEO4J_PORT>
export GRAPH_DB_USERNAME=<GRAPH_DB_USERNAME>
export GRAPH_DB_PASSWORD=<GRAPH_DB_PASSWORD>
export NVIDIA_API_KEY=<NVIDIA_API_KEY>
```

## Using the Python Library


### Setting up config file

First create a config file to set the llms, prompts, and parameters.
Here is an example of the config file:

``` yaml
summarization:
  enable: true
  method: "batch"
  llm:
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    max_tokens: 2048
    temperature: 0.2
    top_p: 0.7
  embedding:
    model: "nvidia/llama-3.2-nv-embedqa-1b-v2"
    base_url: https://integrate.api.nvidia.com/v1
  params:
    batch_size: 5
    batch_max_concurrency: 20
  prompts:
    caption: "Write a concise and clear dense caption for the provided warehouse video, focusing on irregular or hazardous events such as boxes falling, workers not wearing PPE, workers falling, workers taking photographs, workers chitchatting, forklift stuck, etc. Start and end each sentence with a time stamp."
    caption_summarization: "You should summarize the following events of a warehouse in the format start_time:end_time:caption. For start_time and end_time use . to seperate seconds, minutes, hours. If during a time segment only regular activities happen, then ignore them, else note any irregular activities in detail. The output should be bullet points in the format start_time:end_time: detailed_event_description. Don't return anything else except the bullet points."
    summary_aggregation: "You are a warehouse monitoring system. Given the caption in the form start_time:end_time: caption, Aggregate the following captions in the format start_time:end_time:event_description. If the event_description is the same as another event_description, aggregate the captions in the format start_time1:end_time1,...,start_timek:end_timek:event_description. If any two adjacent end_time1 and start_time2 is within a few tenths of a second, merge the captions in the format start_time1:end_time2. The output should only contain bullet points.  Cluster the output into Unsafe Behavior, Operational Inefficiencies, Potential Equipment Damage and Unauthorized Personnel"

chat:
  rag: vector-rag # graph-rag or vector-rag
  params:
    batch_size: 1
  llm:
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    max_tokens: 2048
    temperature: 0.5
  embedding:
    model: "nvidia/llama-3.2-nv-embedqa-1b-v2"
    base_url: https://integrate.api.nvidia.com/v1
  reranker:
    model: "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    base_url: https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking
```

### How to locally deploy Reranker, Embedding, and LLM NIMs

- [Deploying Reranker](https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/deploy)

- [Deploying Embedding](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2/deploy)

- [Deploying LLM](https://build.nvidia.com/meta/llama-3_1-70b-instruct/deploy)

We can then change the base_url in the config file to the local url.

``` yaml
model: meta/llama-3.1-70b-instruct
base_url: http://localhost:8000
```

### Context Manager Setup

Now setup the context manager. Context manager is used to both add
documents and retrieve documents.

``` python
with open("config/config.yaml", mode="r", encoding="utf8") as c:
        config = yaml.safe_load(c)
    ### IF USING VECTOR-RAG
    config["milvus_db_host"] = os.environ["MILVUS_HOST"]
    config["milvus_db_port"] = os.environ["MILVUS_PORT"]

    config["api_key"] = os.environ["NVIDIA_API_KEY"]

    class RequestInfo:
        def __init__(self):
            self.summarize = True
            self.enable_chat = False
            self.is_live = False
            self.uuid = "test_context_manager"
            self.caption_summarization_prompt = (
                "Return the input in it's entirety as is without any changes"
            )
            self.summary_aggregation_prompt = (
                "Combine the conversation into a single summary"
            )
            self.chunk_size = 0
            self.summary_duration = 0
            self.summarize_top_p = None
            self.summarize_temperature = None
            self.summarize_max_tokens = None
            self.chat_top_p = None
            self.chat_temperature = None
            self.chat_max_tokens = None
            self.notification_top_p = None
            self.notification_temperature = None
            self.notification_max_tokens = None
            self.rag_type = config["chat"]["rag"]

    req_info = RequestInfo()

    cm = ContextManager(config=config)
    cm.configure_update(config=config, req_info=req_info)
    ## cm doing work here
    cm.process.stop()
```

### Document Ingestion

Context manager can be used to ingest documents.

``` python
cm.add_doc("User1: I went hiking to Mission Peak") ## Add documents to the context manager
```

## Document Retrieval

To retrieve documents, use the following code as an example:

``` python
question = "Where did the user go hiking?"
result = cm.call(
    {
        "chat": {
            "question": question,
            "is_live": False,
            "is_last": False,
        }
    }
)
logger.info(f"Response {result['chat']['response']}")
```
