import chromadb


class VectorStore:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            "embeddinggemma2_multimodal", metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def add(self, rows):
        self.collection.upsert(ids=[r["id"] for r in rows], embeddings=[r["embedding"] for r in rows],
                               documents=[r["text"] for r in rows], metadatas=[r["metadata"] for r in rows])

    def search(self, embedding, limit=12, modality=None):
        if not self.collection.count():
            return []
        args = {"query_embeddings": [embedding], "n_results": min(limit, self.collection.count()),
                "include": ["documents", "metadatas", "distances"]}
        if modality:
            args["where"] = {"modality": modality}
        found = self.collection.query(**args)
        return [{"id": identifier, "rank": i + 1, "similarity": float(1 - found["distances"][0][i]),
                 "text": found["documents"][0][i], **found["metadatas"][0][i],
                 "explanation": "Ranked by cosine similarity of native embeddings; this is not a probability."}
                for i, identifier in enumerate(found["ids"][0])]

    def keyword(self, query):
        tokens = set(query.lower().split())
        data = self.collection.get(where={"modality": "text"}, include=["documents", "metadatas"])
        results = []
        for identifier, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            count = sum(text.lower().count(token) for token in tokens)
            if count:
                results.append({"id": identifier, "text": text, "keyword_hits": count, **meta})
        return sorted(results, key=lambda row: row["keyword_hits"], reverse=True)[:12]

    def files(self):
        data = self.collection.get(include=["metadatas", "documents"])
        files = {}
        for meta, text in zip(data["metadatas"], data["documents"]):
            key = meta["file_id"]
            if key not in files:
                files[key] = {**meta, "text": text, "vectors": 0}
            files[key]["vectors"] += 1
        return list(files.values())
