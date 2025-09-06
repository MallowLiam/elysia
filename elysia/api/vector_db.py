from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import os


class VectorDB(ABC):
    """Abstract interface for vector databases.

    Implementations should provide simple collection management and basic insert/search
    primitives that are easy to adapt inside tools or services.
    """

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """Establish a client connection.

        Implementations should accept either a `url` or `host`/`port` (plus optional
        `api_key`) keyword arguments. Implementations may also read from environment
        variables as a convenience.
        """

    @abstractmethod
    def create_collection(self, name: str, vector_size: int) -> None:
        """Create a collection with the given vector size if it does not exist."""

    @abstractmethod
    def insert_vectors(
        self,
        collection: str,
        vectors: Iterable[Iterable[float]],
        payloads: Iterable[Dict[str, Any]],
    ) -> None:
        """Insert vectors with associated payloads into the collection."""

    @abstractmethod
    def search_vectors(
        self, collection: str, query_vector: Iterable[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for nearest vectors returning payloads and scores."""


class QdrantVectorDB(VectorDB):
    """Qdrant-backed implementation of VectorDB.

    Relies on `qdrant-client`. Import is done lazily to keep optional deps light.
    """

    def __init__(self) -> None:
        self.client = None  # type: ignore[attr-defined]

    def _require_client(self):
        if self.client is None:
            raise RuntimeError(
                "Qdrant client is not connected. Call connect() before using the instance."
            )
        return self.client

    def connect(self, **kwargs) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "qdrant-client is required for QdrantVectorDB. Install with `pip install qdrant-client`."
            ) from e

        if client := kwargs.get("client"):
            self.client = client
            return

        url: Optional[str] = kwargs.get("url") or os.getenv("QDRANT_URL")
        api_key: Optional[str] = kwargs.get("api_key") or os.getenv("QDRANT_API_KEY")
        prefer_grpc: bool = bool(kwargs.get("prefer_grpc", False))

        if url:
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        else:
            host: str = kwargs.get("host", "localhost")
            port: int = int(kwargs.get("port", 6333))
            self.client = QdrantClient(
                host=host, port=port, api_key=api_key, prefer_grpc=prefer_grpc
            )

    def create_collection(self, name: str, vector_size: int) -> None:
        client = self._require_client()
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        # idempotent create
        try:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except Exception:
            # Best-effort: collection may already exist
            pass

    def insert_vectors(
        self,
        collection: str,
        vectors: Iterable[Iterable[float]],
        payloads: Iterable[Dict[str, Any]],
    ) -> None:
        client = self._require_client()
        from qdrant_client.models import PointStruct  # type: ignore

        pts = []
        for i, (v, p) in enumerate(zip(vectors, payloads)):
            pts.append(PointStruct(id=i, vector=list(v), payload=p))
        if pts:
            client.upsert(collection_name=collection, points=pts)

    def search_vectors(
        self, collection: str, query_vector: Iterable[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        client = self._require_client()
        results = client.search(
            collection_name=collection, query_vector=list(query_vector), limit=limit
        )
        return [
            {"id": getattr(r, "id", None), "payload": r.payload, "score": r.score}
            for r in results
        ]


def create_vector_db(db_type: str, **kwargs) -> VectorDB:
    """Factory for VectorDB implementations.

    Supported types: "qdrant". Additional backends can be wired in here.
    """

    db_type = (db_type or "").strip().lower()
    if db_type == "qdrant":
        db = QdrantVectorDB()
        db.connect(**kwargs)
        return db
    raise ValueError(f"Unsupported vector db type: {db_type}")


def get_vector_db_from_env() -> VectorDB:
    """Create a VectorDB using environment configuration.

    - VECTOR_DB_TYPE: "weaviate" | "qdrant" (currently only qdrant supported here)
    - QDRANT_URL / QDRANT_API_KEY: used when VECTOR_DB_TYPE=qdrant
    """

    backend = os.getenv("VECTOR_DB_TYPE", "qdrant")
    if backend.lower() == "qdrant":
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        return create_vector_db("qdrant", url=url, api_key=api_key)
    raise ValueError(
        f"Unsupported VECTOR_DB_TYPE '{backend}'. Only 'qdrant' is supported by this module."
    )

