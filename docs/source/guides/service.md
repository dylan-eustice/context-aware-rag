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

# Running the Service

### Prerequisites

Either have a running Milvus or Neo4j instance depending on the
configuration you want to use (milvus for Vector-RAG or neo4j for
Graph-RAG).

#### Export the following envs as needed

``` bash
# Required if using NVIDIA Endpoints
NVIDIA_API_KEY=<NVIDIA_API_KEY>

# If using OpenAI Endpoints
OPENAI_API_KEY=<OPENAI_API_KEY>

# For Vector-RAG
MILVUS_HOST=<HOST>
MILVUS_PORT=<MILVUS_PORT>

# For Graph-RAG
GRAPH_DB_URI=bolt://<HOST>:<NEO4J_PORT>
GRAPH_DB_USERNAME=<USERNAME>
GRAPH_DB_PASSWORD=<PASSWORD>
```

#### Start the ingestion service

``` bash
export VIA_CTX_RAG_ENABLE_RET=false
uvicorn service.service:app --host 0.0.0.0 --port <INGEST_PORT>
```

#### Start the retrieval service

``` bash
export VIA_CTX_RAG_ENABLE_RET=true
uvicorn service.service:app --host 0.0.0.0 --port <RETRIEVAL_PORT>
```
