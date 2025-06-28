# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""adv_graph_retrieval.py: File contains AdvGraphRetrieval class"""

from datetime import datetime, timezone
from typing import List, Dict, Any
import json

from langchain_core.documents import Document
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.utils.utils import remove_lucene_chars, remove_think_tags
from vss_ctx_rag.functions.rag.graph_rag.constants import (
    CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
    QUESTION_TRANSFORM_TEMPLATE,
    VECTOR_SEARCH_TOP_K,
    CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
)
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import HumanMessage


class AdvGraphRetrieval:
    def __init__(self, llm, graph: Neo4jGraphDB, top_k=None, max_retries=None):
        logger.info("Initializing AdvGraphRetrieval")
        self.chat_llm = llm
        self.graph_db = graph
        self.top_k = top_k
        self.max_retries = max_retries if max_retries else 3
        self.vector_retriever = Neo4jVector.from_existing_index(
            embedding=self.graph_db.embeddings,
            index_name="vector",
            graph=self.graph_db.graph_db,
        ).as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.top_k or VECTOR_SEARCH_TOP_K,
                "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
            },
        )
        self.doc_retriever = self.create_document_retriever_chain()
        logger.info(f"Initialized with top_k={top_k}")

    def _format_relative_time(self, timestamp: float) -> str:
        """Format timestamp relative to current time in seconds"""
        if timestamp is None:
            return "unknown time"

        now = datetime.now(timezone.utc).timestamp()
        diff = now - timestamp
        return f"{int(diff)} seconds ago"

    def create_document_retriever_chain(self):
        with TimeMeasure("GraphRetrieval/CreateDocRetChain", "blue"):
            try:
                logger.info("Starting to create document retriever chain")

                query_transform_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", QUESTION_TRANSFORM_TEMPLATE),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )

                output_parser = StrOutputParser()

                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.graph_db.embeddings,
                    similarity_threshold=CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
                )
                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[embeddings_filter]
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=pipeline_compressor,
                    base_retriever=self.vector_retriever,
                )
                query_transforming_retriever_chain = RunnableBranch(
                    (
                        lambda x: len(x.get("messages", [])) == 1,
                        (lambda x: x["messages"][-1].content)
                        | output_parser
                        | remove_lucene_chars
                        | compression_retriever,
                    ),
                    query_transform_prompt
                    | self.chat_llm
                    | output_parser
                    | remove_think_tags
                    | remove_lucene_chars
                    | compression_retriever,
                ).with_config(run_name="chat_retriever_chain")

                logger.info("Successfully created document retriever chain")
                return query_transforming_retriever_chain

            except Exception as e:
                logger.error(
                    f"Error creating document retriever chain: {e}", exc_info=True
                )
                raise

    async def get_all_entity_types(self) -> List[str]:
        """Fetch all distinct entity types (node labels) from the Neo4j database"""
        logger.info("Fetching all entity types from Neo4j")

        query = """
        CALL db.labels()
        YIELD label
        RETURN DISTINCT label
        ORDER BY label
        """

        try:
            result = await self.graph_db.arun_cypher_query(query)
            entity_types = [record["label"] for record in result]
            logger.info(f"Found {len(entity_types)} entity types: {entity_types}")
            return entity_types
        except Exception as e:
            logger.error(f"Error fetching entity types: {e}")
            return []

    async def get_all_stream_ids(self) -> List[str]:
        """Fetch all distinct stream_id values from the Neo4j database"""
        logger.info("Fetching all stream_ids from Neo4j")

        query = """
        MATCH (n:Chunk)
        WHERE n.stream_id IS NOT NULL
        RETURN DISTINCT n.stream_id as stream_id
        ORDER BY stream_id
        """

        try:
            result = await self.graph_db.arun_cypher_query(query)
            stream_ids = [record["stream_id"] for record in result]
            logger.info(f"Found {len(stream_ids)} stream_ids: {stream_ids}")
            return stream_ids
        except Exception as e:
            logger.error(f"Error fetching stream_ids: {e}")
            return []

    def _build_property_filters(self, properties: Dict) -> str:
        if not properties:
            return ""
        filters = []
        for key, value in properties.items():
            filters.append(f"n.{key} = '{value}'")
        return "WHERE " + " AND ".join(filters)

    async def retrieve_by_entity_type(
        self, entity_type: str, properties: Dict = None
    ) -> List[Dict]:
        """Retrieve nodes of specific type with optional property filters"""
        logger.info(
            f"Retrieving entities of type {entity_type} with properties {properties}"
        )
        query = f"""
        MATCH (n:{entity_type})
        {self._build_property_filters(properties) if properties else ""}
        RETURN n
        LIMIT {self.top_k or 10}
        """
        return await self.graph_db.arun_cypher_query(query)

    async def retrieve_by_relationship(
        self,
        start_type: str,
        relationship: str,
        end_type: str,
        time_range: Dict = None,
    ) -> List[Dict]:
        """Retrieve relationships between node types with optional time filtering"""
        logger.info(
            f"Retrieving relationships {relationship} between "
            f"{start_type} and {end_type}"
        )
        time_filter = ""
        if time_range:
            time_filter = f"""
            WHERE r.start_timestamp >= {time_range.get("start", 0)}
            AND r.end_timestamp <= {time_range.get("end", "infinity")}
            """
            logger.info(f"Added time filter: {time_range}")

        query = f"""
        MATCH (start:{start_type})-[r:{relationship}]->(end:{end_type})
        {time_filter}
        RETURN start, r, end
        LIMIT {self.top_k or 10}
        """
        return await self.graph_db.arun_cypher_query(query)

    async def retrieve_temporal_context(
        self, start_time: float, end_time: float, stream_ids: List[str] = None
    ) -> List[Dict]:
        """Retrieve all events between start and end times, optionally filtered by stream_id"""
        logger.info(f"Retrieving temporal context between {start_time} and {end_time}")
        if stream_ids:
            logger.info(f"Filtering by stream_ids: {stream_ids}")

        result = []
        temporal_filter = ""
        stream_filter = ""

        if start_time is None and end_time is None:
            return result
        if start_time is not None:
            temporal_filter = f"""
            AND toFloat(n.start_time) >= {start_time}
            """
        if end_time is not None:
            temporal_filter = (
                temporal_filter
                + f"""
            AND toFloat(n.end_time) <= {end_time}
            """
            )

        if stream_ids:
            # Create filter for multiple stream_ids
            stream_conditions = [f"n.stream_id = '{stream_id}'" for stream_id in stream_ids]
            stream_filter = f"""
            AND ({' OR '.join(stream_conditions)})
            """

        query = f"""
        MATCH (n: Chunk)
        WHERE n.start_time IS NOT NULL AND n.end_time IS NOT NULL
        {temporal_filter}
        {stream_filter}
        RETURN n
            ORDER BY n.start_time
            LIMIT {self.top_k or 10}
            """
        result = await self.graph_db.arun_cypher_query(query)
        return result

    async def retrieve_semantic_context(
        self,
        question: str,
        start_time: float = None,
        end_time: float = None,
        sort_by: str = None,
        stream_ids: List[str] = None,
    ) -> List[Dict]:
        """Retrieve semantically similar content using vector similarity search"""
        logger.info(
            f"Retrieving semantic context for question: {question} "
            f"between {start_time} and {end_time}"
        )
        if stream_ids:
            logger.info(f"Filtering by stream_ids: {stream_ids}")

        try:
            result = await self.doc_retriever.ainvoke(
                {"messages": [HumanMessage(content=question)]}
            )
            # logger.info(f"Semantic search results raw: {result}")
            processed_results = []
            for doc in result:
                # Filter by stream_id if specified
                if stream_ids:
                    doc_stream_id = doc.metadata.get("stream_id")
                    if doc_stream_id not in stream_ids:
                        logger.debug(f"Skipping document with stream_id '{doc_stream_id}' (not in {stream_ids})")
                        continue

                processed_results.append(
                    {
                        "n": {
                            "text": doc.page_content,
                            "start_time": doc.metadata.get("start_time", ""),
                            "end_time": doc.metadata.get("end_time", ""),
                            "chunkIdx": doc.metadata.get("chunkIdx", ""),
                            "score": doc.state.get("query_similarity_score", 0),
                            "stream_id": doc.metadata.get("stream_id", ""),
                        }
                    }
                )
                if sort_by == "score":
                    processed_results.sort(key=lambda x: x["n"]["score"], reverse=True)
                elif sort_by == "start_time":
                    processed_results.sort(key=lambda x: x["n"]["start_time"])
                elif sort_by == "end_time":
                    processed_results.sort(key=lambda x: x["n"]["end_time"])
                else:
                    processed_results.sort(key=lambda x: x["n"]["score"], reverse=True)
            logger.info(f"Semantic search results: {processed_results}")
            return processed_results
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """Use LLM to analyze question and determine retrieval strategy"""
        logger.info(f"Analyzing question: {question}")
        prompt = f"""Analyze this question and identify key elements for graph database retrieval.
        Question: {question}

        Identify and return as JSON:
        1. Entity types mentioned. Available entity types: {await self.get_all_entity_types()}
        2. Relationships of interest
        3. Time references
            a. start_time: How many seconds in the past to start the time range. If not present, set to None.
            b. end_time: How many seconds in the past to end the time range. If not present, set to None.
        4. Sort by: "start_time" or "end_time" or "score"
        5. Location references
        6. Stream IDs mentioned. Available stream_ids: {await self.get_all_stream_ids()}
        7. Retrieval strategy (similarity, temporal)
            a. similarity: If the question needs to find similar content, return the retrieval strategy as similarity
            b. temporal: If the question is about a specific time range and you can return at least one of the start and end time, then return the strategy as temporal and the start and end time in the time_references field as float or null if not present. Strategy cannot be temporal if both start and end time are not present. The start and end time should be in seconds.

        Example question: "Between 30 seconds and 5 minutes ago, has the dog found the ball?"
        Example response:
        {{\
            "entity_types": ["Dog", "Ball"],\
            "relationships": ["DROPPED", "PICKED_UP"],\
            "time_references": {{\
                "start": 300.0,\
                "end": 30.0\
            }},\
            "sort_by": "start_time", // "start_time" or "end_time" or "score" \
            "location_references": ["backyard"],\
            "stream_ids": [],\
            "retrieval_strategy": "temporal"\
        }}\

        Example question with stream: "Summarize channel 3 over the last 5 minutes."
        Example response:
        {{\
            "entity_types": [],\
            "relationships": [],\
            "time_references": {{\
                "start": 300.0,\
                "end": 0.0\
            }},\
            "sort_by": "start_time",\
            "location_references": [],\
            "stream_ids": ["fm-radio-ch3"],\
            "retrieval_strategy": "temporal"\
        }}\

        NOTE: When setting time references (start_time, end_time), we should focus on 3 classes of time windows:
        1. Window of the last X seconds
            a. Summarize the last half hour -> start_time, end_time = 1800, 0
            b. Look for <topic> over the previous 10 mins -> start_time, end_time = 600, 0
        2. Window between X and Y seconds ago
            a. Look for <topic> between 10 and 20 seconds ago -> start_time, end_time = 20, 10
            b. What was being discussed between 5 minutes and half an hour ago -> start_time, end_time = 1800, 300
        3. Window centered around X seconds ago, with a window of Y seconds (default Y = 300 seconds)
            a. Look for <topic> from 10 minutes ago -> start_time, end_time = 900, 300
            b. What was being discussed and hour ago? Use a 15 minute window. -> start_time, end_time = 4500, 2700

        Output only valid JSON. Do not include any other text.
        """

        # summarize last one hour
        # summarize events between 60 and 400 seconds

        response = await self.chat_llm.ainvoke(prompt)
        logger.info("Question analysis complete")
        # Parse LLM response to get retrieval strategy
        # This is a simplified version - you'd want proper JSON parsing
        return remove_think_tags(response.content)

    async def retrieve_relevant_context(self, question: str) -> List[Document]:
        """Main retrieval method that orchestrates different retrieval strategies"""
        with TimeMeasure("AdvGraphRetrieval/retrieve_context", "blue"):
            logger.info(f"Starting context retrieval for question: {question}")
            analysis_response = await self.analyze_question(question)
            json_start = analysis_response.find("{")
            json_end = analysis_response.rfind("}") + 1

            # Parse the JSON response from the LLM with retries
            logger.info(f"Analysis response: {analysis_response}")
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    analysis = json.loads(analysis_response[json_start:json_end])
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        # Retry getting analysis
                        analysis_response = await self.analyze_question(question)
                        json_start = analysis_response.find("{")
                        json_end = analysis_response.rfind("}") + 1
                        logger.info(
                            f"Retry {retry_count}: New analysis response: {analysis_response}"
                        )
                    else:
                        logger.error("Max retries reached, using default analysis")
                        analysis = {
                            "entity_types": [],
                            "relationships": [],
                            "time_references": {},
                            "location_references": [],
                            "stream_ids": [],
                            "sort_by": "score",
                            "retrieval_strategy": "similarity",
                        }

            # Collect context from multiple retrieval strategies
            contexts = []

            # Get retrieval strategy and parameters from analysis
            strategy = analysis.get("retrieval_strategy", "")
            logger.info(f"Using retrieval strategy: {strategy}")

            time_refs = analysis.get("time_references", {})
            start_time = None
            end_time = None
            stream_ids = analysis.get("stream_ids", [])

            # Temporal context retrieval
            if time_refs:
                tnow = datetime.now(timezone.utc).timestamp()
                start_time = tnow - time_refs.get("start") if time_refs.get("start") is not None else None
                end_time = tnow - time_refs.get("end") if time_refs.get("end") is not None else None

            if strategy == "temporal":
                temporal_data = await self.retrieve_temporal_context(
                    start_time, end_time, stream_ids
                )
                logger.info(f"Temporal Contexts...")
                if temporal_data:
                    contexts.extend(temporal_data)
                    logger.info(f"Retrieved {len(temporal_data)} temporal records")
                else:
                    logger.info("No temporal data found in that time range")
                    return None
            else:  # semantic retrieval
                # Semantic similarity retrieval
                semantic_data = await self.retrieve_semantic_context(
                    question,
                    start_time=start_time,
                    end_time=end_time,
                    sort_by=analysis.get("sort_by", "score"),
                    stream_ids=stream_ids,
                )
                logger.info(f"Semantic Contexts...")
                if semantic_data:
                    contexts.extend(semantic_data)

            logger.info(f"Contexts: {contexts}")

            # Relationship-based retrieval
            relationships = analysis.get("relationships", [])
            for rel in relationships:
                if isinstance(rel, str):
                    # If relationship is specified without types, skip
                    continue
                start_type = rel.get("from")
                end_type = rel.get("to")
                rel_type = rel.get("type")
                if all([start_type, end_type, rel_type]):
                    rel_data = await self.retrieve_by_relationship(
                        start_type, rel_type, end_type
                    )
                    logger.info(f"Relationship Data: {rel_data}")
                    if rel_data:
                        contexts.extend(rel_data)
                        logger.info(
                            f"Retrieved {len(rel_data)} records for "
                            f"relationship {rel_type}"
                        )

            # Convert to Documents
            documents = []
            for ctx in contexts:
                # Convert Neo4j results to Document format
                # Check if ctx has expected structure
                if isinstance(ctx, dict) and "n" in ctx:
                    if "text" in ctx["n"]:
                        doc = Document(
                            page_content=str(ctx["n"].get("text", "")),
                            metadata={
                                "start_time": ctx.get("n", {}).get("start_time", ""),
                                "end_time": ctx.get("n", {}).get("end_time", ""),
                                "stream_id": ctx.get("n", {}).get("stream_id", ""),
                            },
                        )
                        documents.append(doc)

            logger.info(f"Returning {len(documents)} documents")
            return documents
