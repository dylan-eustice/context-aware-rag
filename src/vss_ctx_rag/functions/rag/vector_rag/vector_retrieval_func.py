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

"""graph_rag.py: File contains Function class"""

import os
from pathlib import Path
from re import compile
import traceback
from typing import Optional, List, Dict
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from vss_ctx_rag.base import Function
from vss_ctx_rag.tools.storage.milvus_db import MilvusDBTool
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.utils.globals import DEFAULT_RAG_TOP_K, LLM_TOOL_NAME


class VectorRetrievalFunc(Function):
    """VectorRAG Function"""

    config: dict
    output_parser = StrOutputParser()
    vector_db: MilvusDBTool
    metrics = GraphMetrics()

    def setup(self):
        # Log the entire configuration passed to this function
        logger.info(f"VectorRetrievalFunc setup called")
        logger.info(f"Full _params configuration: {self._params}")
        logger.info(f"Configuration keys available: {list(self._params.keys()) if self._params else 'None'}")

        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.vector_db = self.get_tool("vector_db")
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )
        self.regex_object = compile(r"<(\d+[.]\d+)>")

        # Simplified citation configuration
        logger.info("Attempting to load citations configuration...")
        self.citations_config = self._params.get("citations", {})
        logger.info(f"Raw citations_config retrieved: {self.citations_config}")

        self.citations_enabled = self.citations_config.get("enabled", False)
        self.include_metadata = self.citations_config.get("include_metadata", True)
        self.citation_fields = self.citations_config.get("citation_fields", ["doc_id", "filename"])
        self.citation_template = self.citations_config.get("citation_template", "[{doc_id}] {filename}")
        self.show_snippets = self.citations_config.get("show_snippets", False)
        self.snippet_length = self.citations_config.get("snippet_length", 200)

        # Log citation configuration
        logger.info(f"Citations configuration loaded: {self.citations_config}")
        logger.info(f"Citations enabled: {self.citations_enabled}")
        logger.info(f"Citation template: {self.citation_template}")
        logger.info(f"Show snippets: {self.show_snippets}")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        embeddings_dimension = int(os.environ.get("CA_RAG_EMBEDDINGS_DIMENSION", 1024))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=embeddings_dimension,
            chunk_overlap=0,
            separators=["\n\n", "\n", "\n-", ".", ";", ",", " ", ""],
        )
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, self.vector_db.reranker]
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=self.vector_db.vector_db.as_retriever(
                search_kwargs={"filter": {"doc_type": "caption"}, "k": self.top_k}
            ),
        )
        self.g_semantic_sim_chain = RetrievalQA.from_chain_type(
            llm=self.chat_llm, retriever=self.compression_retriever, return_source_documents=True
        )

    def format_citation(self, doc_metadata: Dict, citation_id: int) -> str:
        """Format a single citation based on the configuration template."""
        logger.debug(f"Formatting citation {citation_id} with metadata: {doc_metadata}")
        try:
            # Add citation ID to metadata for template formatting
            metadata_with_id = {**doc_metadata, "citation_id": citation_id}

            # Format using the template, with fallback values for missing fields
            formatted_citation = self.citation_template
            for field in self.citation_fields:
                placeholder = f"{{{field}}}"
                value = metadata_with_id.get(field, f"Unknown_{field}")
                formatted_citation = formatted_citation.replace(placeholder, str(value))

            logger.debug(f"Formatted citation: {formatted_citation}")
            return formatted_citation
        except Exception as e:
            logger.warning(f"Error formatting citation: {e}")
            return f"[{citation_id}] Document"

    def format_citations_display(self, citations: List[Dict], retrieved_docs) -> str:
        """Format citations for inline display."""
        logger.info(f"Formatting citations display. Citations count: {len(citations) if citations else 0}")
        logger.info(f"Citations enabled: {self.citations_enabled}")

        if not citations or not self.citations_enabled:
            logger.info("No citations to display or citations disabled")
            return ""

        citation_text = "\n**Sources:**\n"
        logger.info(f"Processing {len(citations)} citations")

        for i, citation in enumerate(citations):
            formatted_citation = self.format_citation(citation, i + 1)

            if self.show_snippets and i < len(retrieved_docs):
                snippet = retrieved_docs[i].page_content[:self.snippet_length]
                if len(retrieved_docs[i].page_content) > self.snippet_length:
                    snippet += "..."
                citation_text += f"- **{formatted_citation}** - *\"{snippet}\"*\n"
                logger.debug(f"Added snippet for citation {i + 1}: {snippet[:50]}...")
            else:
                citation_text += f"- **{formatted_citation}**\n"
        citation_text += "\n - - - - - \n"

        logger.info(f"Final citation text length: {len(citation_text)}")
        logger.debug(f"Citation text preview: {citation_text[:200]}...")
        return citation_text

    def extract_citations_from_docs(self, retrieved_docs) -> List[Dict]:
        """Extract citation information from retrieved documents."""
        logger.info(f"Extracting citations from {len(retrieved_docs)} retrieved documents")
        citations = []

        for i, doc in enumerate(retrieved_docs):
            logger.debug(f"Processing document {i}: has metadata: {hasattr(doc, 'metadata')}")
            if hasattr(doc, 'metadata'):
                logger.info(f"Document {i} metadata keys: {list(doc.metadata.keys())}")
                logger.info(f"Document {i} full metadata: {doc.metadata}")

            if hasattr(doc, 'metadata') and self.include_metadata:
                citation_info = {}
                for field in self.citation_fields:
                    citation_info[field] = doc.metadata.get(field, f"Unknown_{field}")
                citations.append(citation_info)
                logger.debug(f"Added citation {i}: {citation_info}")
            else:
                # Basic citation info if metadata is not available
                basic_citation = {
                    "doc_id": len(citations) + 1,
                    "filename": "Document",
                    "timestamp": "Unknown"
                }
                citations.append(basic_citation)
                logger.debug(f"Added basic citation {i}: {basic_citation}")

        logger.info(f"Extracted {len(citations)} citations total")
        return citations

    async def acall(self, state: dict):
        """QnA function call"""
        if self.log_dir:
            with TimeMeasure("VectorRAG/aprocess-doc/metrics_dump", "yellow"):
                log_path = Path(self.log_dir).joinpath("vector_rag_metrics.json")
                self.metrics.dump_json(log_path.absolute())
        try:
            logger.debug("Running qna with question: %s", state["question"])
            with TimeMeasure("VectorRAG/retrieval", "red"):
                semantic_search_answer = await self.g_semantic_sim_chain.ainvoke(
                    state["question"]
                )
                logger.debug(
                    "Semantic search response: %s", semantic_search_answer["result"]
                )
                logger.info(f"Semantic search keys: {list(semantic_search_answer.keys())}")

                response = semantic_search_answer["result"]
                response = self.regex_object.sub(r"\g<1>", response)

                # Add citations if enabled
                logger.info(f"Checking for citations. Enabled: {self.citations_enabled}")
                if self.citations_enabled and "source_documents" in semantic_search_answer:
                    retrieved_docs = semantic_search_answer["source_documents"]
                    logger.info(f"Found {len(retrieved_docs)} source documents")

                    citations = self.extract_citations_from_docs(retrieved_docs)
                    citation_display = self.format_citations_display(citations, retrieved_docs)

                    if citation_display:
                        logger.info("Adding citations to response")
                        response = citation_display + response
                    else:
                        logger.warning("Citation display is empty")

                    # Store citations in state for potential UI use
                    state["citations"] = citations
                    state["retrieved_docs"] = len(retrieved_docs)
                    logger.info(f"Stored {len(citations)} citations in state")
                else:
                    if not self.citations_enabled:
                        logger.info("Citations are disabled")
                    if "source_documents" not in semantic_search_answer:
                        logger.warning("No source_documents in semantic search response")

                state["response"] = response
                logger.info(f"Final response length: {len(response)}")

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in QA %s", str(e))
            state["response"] = "That didn't work. Try another question."
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        pass

    async def areset(self, expr):
        self.metrics.reset()
        self.vector_db.drop_data("pk > 0")
        await asyncio.sleep(0.01)
