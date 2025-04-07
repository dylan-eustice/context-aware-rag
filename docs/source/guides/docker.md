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

# Running the Standalone Docker Service

#### neo4j

``` bash
docker run -d \
  --name neo4j \
  -p <NEO4J_PORT>:7687 \
  -e NEO4J_AUTH=<GRAPH_DB_USERNAME>/<GRAPH_DB_PASSWORD> \
  neo4j:5.26.4
```

#### milvus

``` bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh


bash standalone_embed.sh start
```

This will start the milvus service by default on port 19530.

#### Export Environment Variables

``` bash
MILVUS_HOST=<HOST>
MILVUS_PORT=<MILVUS_PORT>

GRAPH_DB_URI=bolt://<HOST>:<NEO4J_PORT>
GRAPH_DB_USERNAME=<GRAPH_DB_USERNAME>
GRAPH_DB_PASSWORD=<GRAPH_DB_PASSWORD>
```

## Build the vss_ctx_rag image

Make sure you are in the project root directory.

``` bash
make -C docker build
```

## Run the data ingestion service

``` bash
make -C docker start_in
```

## Run the data retrieval service

``` bash
make -C docker start_ret
```
