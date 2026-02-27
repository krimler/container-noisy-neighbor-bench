"""
RAG Knowledge Base using ChromaDB.
Stores container configs, runbooks, and operational knowledge.
Retrieves relevant context for the LLM analyzer.
"""
import json
import chromadb
from typing import List, Dict


class ConfigKnowledgeBase:
    def __init__(self, knowledge_path: str = "knowledge/container_configs.json"):
        self.client = chromadb.Client()  # In-memory, no persistence needed for POC
        self.collection = self.client.get_or_create_collection(
            name="config_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        self._load_knowledge(knowledge_path)

    def _load_knowledge(self, path: str):
        """Load config documents into ChromaDB."""
        with open(path, "r") as f:
            docs = json.load(f)

        if self.collection.count() > 0:
            return  # Already loaded

        ids = [d["id"] for d in docs]
        documents = [d["content"] for d in docs]
        metadatas = [
            {"type": d["type"], "container": d["container"]} for d in docs
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[RAG] Loaded {len(docs)} knowledge documents")

    def query(self, query_text: str, n_results: int = 4) -> List[Dict]:
        """Retrieve relevant config knowledge for a query."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return retrieved

    def get_container_config(self, container_name: str) -> str:
        """Get the specific config knowledge for a container."""
        results = self.collection.get(
            where={"container": container_name},
        )
        if results["documents"]:
            return "\n\n".join(results["documents"])
        return f"No configuration knowledge found for {container_name}"

    def get_runbooks(self) -> str:
        """Get all operational runbooks."""
        results = self.collection.get(
            where={"type": "operational_runbook"},
        )
        if results["documents"]:
            return "\n\n".join(results["documents"])
        return "No runbooks available"

    def query_for_diagnosis(self, victim: str, suspect: str) -> str:
        """Build a focused context retrieval for a noisy neighbor scenario."""
        # Get victim config
        victim_ctx = self.get_container_config(victim)
        # Get suspect config
        suspect_ctx = self.get_container_config(suspect)
        # Get relevant runbooks
        runbook_ctx = self.get_runbooks()
        # Get technical knowledge
        tech_results = self.collection.get(
            where={"type": "technical_knowledge"},
        )
        tech_ctx = "\n\n".join(tech_results["documents"]) if tech_results["documents"] else ""

        return f"""=== VICTIM CONTAINER CONFIG ===
{victim_ctx}

=== SUSPECT CONTAINER CONFIG ===
{suspect_ctx}

=== OPERATIONAL RUNBOOKS ===
{runbook_ctx}

=== TECHNICAL KNOWLEDGE ===
{tech_ctx}"""
