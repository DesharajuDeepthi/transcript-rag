"""In-memory stand-ins for Chroma and the Anthropic client (no keys, no network)."""

from types import SimpleNamespace


def _match(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(_match(meta, w) for w in where["$or"])
    (key, cond), = where.items()
    (op, val), = cond.items()
    v = meta.get(key)
    return {
        "$eq": lambda: v == val, "$ne": lambda: v != val, "$in": lambda: v in val,
        "$nin": lambda: v not in val, "$gte": lambda: v is not None and v >= val,
        "$lte": lambda: v is not None and v <= val, "$gt": lambda: v is not None and v > val,
        "$lt": lambda: v is not None and v < val,
    }[op]()


class FakeCollection:
    """Chroma subset. 'Similarity' = share of query words present in the chunk.
    fail_upsert_after=N makes the N+1-th upsert call raise (crash simulation)."""

    def __init__(self, rows=None, fail_upsert_after=None):
        self.store: dict[str, tuple[str, dict]] = {cid: (t, m) for cid, t, m in (rows or [])}
        self.queries = []
        self.fail_upsert_after = fail_upsert_after
        self.upserts = 0

    @property
    def rows(self):
        return [(cid, t, m) for cid, (t, m) in self.store.items()]

    def count(self):
        return len(self.store)

    def query(self, query_texts, n_results, where=None, include=None):
        self.queries.append(where)
        q = {w.strip("?,.'").lower() for w in query_texts[0].split()}
        scored = []
        for cid, (text, meta) in self.store.items():
            if not _match(meta, where):
                continue
            words = {w.strip("?,.'").lower() for w in text.split()}
            overlap = len(q & words)
            scored.append((1 - overlap / (len(q) or 1), cid, text, meta))
        scored.sort(key=lambda x: (x[0], x[1]))
        top = scored[:n_results]
        return {"ids": [[s[1] for s in top]], "documents": [[s[2] for s in top]],
                "metadatas": [[s[3] for s in top]], "distances": [[s[0] for s in top]]}

    def get(self, ids=None, where=None, include=None, limit=None):
        rows = [(c, t, m) for c, (t, m) in self.store.items()
                if (ids is None or c in set(ids)) and _match(m, where)][:limit]
        return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows],
                "metadatas": [r[2] for r in rows]}

    def upsert(self, ids, documents, metadatas):
        if self.fail_upsert_after is not None and self.upserts >= self.fail_upsert_after:
            raise ConnectionError("simulated vector store outage")
        self.upserts += 1
        for cid, d, m in zip(ids, documents, metadatas):
            self.store[cid] = (d, dict(m))

    add = upsert

    def delete(self, ids=None, where=None):
        for cid in [c for c, (t, m) in self.store.items()
                    if (ids is None or c in set(ids)) and (where is None or _match(m, where))]:
            del self.store[cid]

    def update(self, ids, metadatas=None, documents=None):
        for i, cid in enumerate(ids):
            text, meta = self.store[cid]
            self.store[cid] = (documents[i] if documents else text, dict(metadatas[i]) if metadatas else meta)


class FakeLLM:
    """
    reply: a string, or a list of (text, stop_reason) returned in order (last one repeats).
    text=None simulates a response with only a thinking block (no visible text).
    """

    def __init__(self, reply="Ravi owns the log inventory (SOC 2 Audit Prep, 2026-02-10) [1]."):
        self.calls = []
        self.replies = reply if isinstance(reply, list) else [(reply, "end_turn")]
        self.messages = self

    def create(self, **kwargs):
        text, stop = self.replies[min(len(self.calls), len(self.replies) - 1)]
        self.calls.append(kwargs)
        blocks = [SimpleNamespace(type="thinking", thinking="...")]
        if text is not None:
            blocks.append(SimpleNamespace(type="text", text=text))
        return SimpleNamespace(content=blocks, stop_reason=stop,
                               usage=SimpleNamespace(input_tokens=1200, output_tokens=80))
