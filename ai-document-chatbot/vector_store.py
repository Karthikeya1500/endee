"""
vector_store.py — Endee Vector Database Client

This module wraps Endee's REST API to provide a simple Python interface
for creating indexes, inserting document embeddings, and performing
semantic similarity search.

How Vector Similarity Search Works:
====================================
1. Text is converted into dense vectors (arrays of floats) using an
   embedding model. Semantically similar texts produce vectors that
   are close together in high-dimensional space.

2. When a user asks a question, the question is also converted into
   a vector using the same embedding model.

3. The vector database finds the stored vectors that are closest to
   the query vector using cosine similarity — a measure of the angle
   between two vectors. A cosine similarity of 1.0 means identical
   direction (maximum similarity), while 0.0 means orthogonal
   (no similarity).

4. Endee uses HNSW (Hierarchical Navigable Small World) graphs for
   fast approximate nearest neighbor search, making retrieval
   efficient even with millions of vectors.
"""

import json
import requests
import msgpack


class EndeeVectorStore:
    """
    A Python client for the Endee vector database.

    Handles index creation, vector insertion, and KNN search
    via Endee's HTTP REST API.
    """

    def __init__(self, base_url: str = "http://localhost:8080", auth_token: str = ""):
        """
        Initialize the Endee client.

        Args:
            base_url: The URL where Endee server is running (default: http://localhost:8080)
            auth_token: Optional authentication token (matches NDD_AUTH_TOKEN on server)
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v1"
        self.auth_token = auth_token

    def _headers(self, content_type: str = "application/json") -> dict:
        """Build request headers with optional auth token."""
        headers = {"Content-Type": content_type}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """
        Check if the Endee server is running and healthy.

        Returns:
            True if the server responds successfully, False otherwise.
        """
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------
    def create_index(self, index_name: str, dim: int, space_type: str = "cosine") -> dict:
        """
        Create a new vector index in Endee.

        Args:
            index_name: Name for the index (e.g., "documents")
            dim: Dimensionality of the vectors (must match embedding model output)
            space_type: Distance metric — "cosine", "l2", or "ip"

        Returns:
            Server response as a dict.

        The index uses HNSW with these defaults:
            - M=16: Each node connects to 16 neighbors (balances speed vs accuracy)
            - ef_con=128: Construction search depth (higher = better graph quality)
            - precision="float32": Full precision storage for accuracy
        """
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "M": 16,
            "ef_con": 128,
            "precision": "float32",
        }
        resp = requests.post(
            f"{self.api_url}/index/create",
            headers=self._headers(),
            json=payload,
            timeout=10,
        )
        return {"status_code": resp.status_code, "message": resp.text}

    def index_exists(self, index_name: str) -> bool:
        """Check if an index already exists on the server."""
        try:
            resp = requests.get(
                f"{self.api_url}/index/{index_name}/info",
                headers=self._headers(),
                timeout=5,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def delete_index(self, index_name: str) -> dict:
        """Delete an existing index."""
        resp = requests.delete(
            f"{self.api_url}/index/{index_name}/delete",
            headers=self._headers(),
            timeout=10,
        )
        return {"status_code": resp.status_code, "message": resp.text}

    # ------------------------------------------------------------------
    # Vector Operations
    # ------------------------------------------------------------------
    def insert_vectors(self, index_name: str, vectors: list[dict]) -> dict:
        """
        Insert vectors with metadata into the index.

        Args:
            index_name: Target index name
            vectors: List of dicts, each containing:
                - id (str): Unique identifier for the vector
                - vector (list[float]): The embedding vector
                - meta (str): Associated text content (stored as metadata)

        Each vector represents a chunk of the original document.
        The 'meta' field stores the raw text so we can retrieve it
        during search and use it as context for answering questions.
        """
        resp = requests.post(
            f"{self.api_url}/index/{index_name}/vector/insert",
            headers=self._headers(),
            json=vectors,
            timeout=30,
        )
        return {"status_code": resp.status_code, "message": resp.text}

    def search(self, index_name: str, query_vector: list[float], k: int = 5) -> list[dict]:
        """
        Perform K-nearest neighbor search to find the most similar vectors.

        Args:
            index_name: Index to search in
            query_vector: The embedding of the user's question
            k: Number of top results to return (default: 5)

        Returns:
            List of results, each containing:
                - id: The vector/chunk identifier
                - similarity: Cosine similarity score (0.0 to 1.0)
                - meta: The original text chunk

        How it works:
            1. The query vector is compared against all stored vectors
            2. Endee's HNSW index efficiently finds approximate nearest neighbors
            3. Results are ranked by cosine similarity (highest = most relevant)
            4. Top K results are returned
        """
        search_body = {
            "vector": query_vector,
            "k": k,
            "ef": 128,  # Search depth — higher values improve accuracy at the cost of speed
            "include_vectors": False,  # We only need the text metadata, not the vectors
        }
        resp = requests.post(
            f"{self.api_url}/index/{index_name}/search",
            headers=self._headers(),
            json=search_body,
            timeout=10,
        )

        # Endee returns search results encoded as MessagePack (binary format).
        # The response is a list of lists, where each inner list is:
        # [similarity, id, meta_bytes, filter_str, norm, vector_list]
        raw = msgpack.unpackb(resp.content, raw=False)

        results = []
        for item in raw:
            similarity = item[0]    # float — cosine similarity score
            vec_id = item[1]        # str — chunk identifier
            meta_value = item[2]    # bytes — original text chunk

            # Meta is returned as bytes — decode to string
            if isinstance(meta_value, bytes):
                meta_value = meta_value.decode("utf-8", errors="replace")

            results.append({
                "id": vec_id,
                "similarity": similarity,
                "meta": meta_value,
            })

        return results
