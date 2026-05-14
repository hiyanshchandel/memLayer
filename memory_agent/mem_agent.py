from memory_blob.definition import MemoryBlob
from episodic.memory_manager import EpisodicMemoryManager
from semantic.sem_mem_man import GraphMemoryManager
from embeddings import get_embedding
from config import GRAPH_CONTEXT_CACHE_LIMIT, GRAPH_CONTEXT_ENTITY_LIMIT, GRAPH_PUSH_MIN_INTERVAL_SECONDS, EPISODIC_RETRIEVAL_FALLBACK_THRESHOLDS
from queue import Queue
import threading
import time
import re

class MemoryAgent:
    _graph_queue = Queue()
    _graph_worker_started = False
    _graph_worker_lock = threading.Lock()
    _graph_context_cache = {}
    _graph_context_cache_lock = threading.Lock()
    _query_stopwords = {
        "a", "an", "and", "are", "as", "be", "did", "do", "does", "for", "from", "give",
        "go", "had", "has", "have", "how", "i", "in", "is", "it", "me", "my", "of",
        "on", "or", "tell", "that", "the", "their", "them", "there", "these", "they",
        "this", "those", "to", "was", "were", "what", "when", "where", "which", "who",
        "whom", "whose", "why", "with", "would", "you", "your", "about", "say", "said",
        "says", "asking", "ask", "asked", "show", "shown", "please", "remember", "recall",
    }
    _query_acronyms = {
        "api": "application programming interface",
        "db": "database",
        "etl": "extract transform load",
        "gpu": "graphics processing unit",
        "llm": "large language model",
        "ml": "machine learning",
        "nlp": "natural language processing",
        "rag": "retrieval augmented generation",
        "sql": "structured query language",
        "ui": "user interface",
    }

    def __init__(self):
        self.episodic_manager = EpisodicMemoryManager()
        self.graph_manager = GraphMemoryManager()
        self._ensure_graph_worker()

    @classmethod
    def _ensure_graph_worker(cls):
        with cls._graph_worker_lock:
            if cls._graph_worker_started:
                return

            worker = threading.Thread(target=cls._graph_worker_loop, daemon=True)
            worker.start()
            cls._graph_worker_started = True

    @classmethod
    def _graph_worker_loop(cls):
        last_push_started = 0.0

        while True:
            memory_blob = cls._graph_queue.get()
            try:
                graph_manager = GraphMemoryManager()
                context_key = cls._graph_context_key(memory_blob)
                known_entities = cls._select_known_entities(context_key)

                elapsed_since_last = time.perf_counter() - last_push_started
                sleep_seconds = max(0.0, GRAPH_PUSH_MIN_INTERVAL_SECONDS - elapsed_since_last)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

                last_push_started = time.perf_counter()
                graph_data = graph_manager.push_to_graphdb(memory_blob, known_entities=known_entities)
                cls._update_graph_context(context_key, graph_data)
            except Exception as exc:
                print(f"[Timing] graph_push_error={exc}")
            finally:
                cls._graph_queue.task_done()

    @classmethod
    def _graph_context_key(cls, memory_blob):
        tags = getattr(memory_blob, "tags", {}) or {}
        source_name = str(tags.get("source_name", "")).strip().lower()
        source_type = str(tags.get("source_type", "")).strip().lower()
        memory_type = str(getattr(memory_blob, "memory_type", "") or "").strip().lower()

        if source_name:
            return f"{source_type}:{source_name}" if source_type else source_name

        return source_type or memory_type or "default"

    @classmethod
    def _normalize_graph_name(cls, value):
        return re.sub(r"[^a-z0-9]+", "", str(value).lower()) if value else ""

    @classmethod
    def _score_graph_entity(cls, record):
        label = record.get("label", "")
        count = record.get("count", 0)
        relation_count = record.get("relation_count", 0)
        label_boosts = {
            "Person": 5,
            "Organization": 4,
            "Location": 4,
            "Project": 4,
            "Technology": 3,
            "Concept": 2,
            "Document": 2,
            "Event": 2,
            "Algorithm": 2,
            "Attribute": 1,
        }

        return (count * 4) + (relation_count * 2) + label_boosts.get(label, 0)

    @classmethod
    def _select_known_entities(cls, context_key):
        with cls._graph_context_cache_lock:
            cache = cls._graph_context_cache.get(context_key, {})
            records = list(cache.values())

        records.sort(key=lambda item: (cls._score_graph_entity(item), item.get("last_seen", 0)), reverse=True)
        shortlist = []
        for record in records[:GRAPH_CONTEXT_ENTITY_LIMIT]:
            shortlist.append({
                "name": record.get("name"),
                "label": record.get("label"),
                "count": record.get("count", 0),
            })

        return shortlist

    @classmethod
    def _update_graph_context(cls, context_key, graph_data):
        entities = graph_data.get("entities", []) if graph_data else []
        relationships = graph_data.get("relationships", []) if graph_data else []

        relation_counts = {}
        for relationship in relationships:
            relation_counts[relationship.get("start_id")] = relation_counts.get(relationship.get("start_id"), 0) + 1
            relation_counts[relationship.get("end_id")] = relation_counts.get(relationship.get("end_id"), 0) + 1

        with cls._graph_context_cache_lock:
            cache = cls._graph_context_cache.setdefault(context_key, {})

            for entity in entities:
                entity_name = entity.get("name")
                entity_label = entity.get("label")
                entity_id = entity.get("id")
                if not entity_name or not entity_label or not entity_id:
                    continue

                entity_key = cls._normalize_graph_name(entity_name)
                if not entity_key:
                    continue

                record = cache.setdefault(entity_key, {
                    "name": entity_name,
                    "label": entity_label,
                    "count": 0,
                    "relation_count": 0,
                    "last_seen": 0,
                })

                record["name"] = entity_name
                record["label"] = entity_label
                record["count"] = record.get("count", 0) + 1
                record["relation_count"] = record.get("relation_count", 0) + relation_counts.get(entity_id, 0)
                record["last_seen"] = time.time()

            if len(cache) > GRAPH_CONTEXT_CACHE_LIMIT:
                sorted_records = sorted(cache.items(), key=lambda item: (cls._score_graph_entity(item[1]), item[1].get("last_seen", 0)), reverse=True)
                trimmed_cache = dict(sorted_records[:GRAPH_CONTEXT_CACHE_LIMIT])
                cls._graph_context_cache[context_key] = trimmed_cache

    def _normalize_query(self, query):
        return " ".join(query.strip().split()).lower()

    def _expand_query_phrase(self, phrase):
        expanded_tokens = []
        for token in re.findall(r"[a-z0-9']+", str(phrase).lower()):
            if token in self._query_stopwords:
                continue

            expansion = self._query_acronyms.get(token)
            if expansion:
                expanded_tokens.extend(expansion.split())
            else:
                expanded_tokens.append(token)

        return " ".join(expanded_tokens).strip()

    def _rewrite_query_for_retrieval(self, query):
        normalized_query = self._normalize_query(str(query or ""))
        if not normalized_query:
            return normalized_query

        stripped_query = re.sub(r"[?.!]+$", "", normalized_query).strip()
        pattern_templates = (
            (r"^(?:what|who|which|where|when|why|how)\s+did\s+i\s+(?:say|mention|ask)\s+about\s+(?P<topic>.+)$", "user discussed {topic}"),
            (r"^(?:what|who|which|where|when|why|how)\s+do\s+i\s+know\s+about\s+(?P<topic>.+)$", "user knows about {topic}"),
            (r"^(?:tell|show|give)\s+me\s+about\s+(?P<topic>.+)$", "{topic}"),
            (r"^(?:what|who|which|where|when|why|how)\s+(?:is|are)\s+(?P<topic>.+)$", "{topic}"),
            (r"^(?:remember|recall)\s+(?P<topic>.+)$", "{topic}"),
            (r"^(?:what|who|which|where|when|why|how)\s+(?:was|were)\s+(?P<topic>.+)$", "{topic}"),
        )

        for pattern, template in pattern_templates:
            match = re.match(pattern, stripped_query)
            if match:
                topic = self._expand_query_phrase(match.group("topic"))
                if topic:
                    return template.format(topic=topic)

        rewritten = self._expand_query_phrase(stripped_query)
        return rewritten or stripped_query

    def _episodic_confidence_from_threshold(self, threshold):
        if threshold is None:
            return {"label": "unbounded", "score": 0.0, "fallback_used": True}

        if threshold >= 0.82:
            return {"label": "high", "score": 0.95, "fallback_used": False}
        if threshold >= 0.65:
            return {"label": "medium", "score": 0.75, "fallback_used": False}
        if threshold >= 0.5:
            return {"label": "guarded", "score": 0.55, "fallback_used": False}
        return {"label": "low", "score": 0.35, "fallback_used": False}

    def _summarize_retrieval_confidence(self, query_runs):
        if not query_runs:
            return {"label": "unknown", "score": 0.0, "fallback_used": False}

        fallback_used = any(run.get("fallback_used") for run in query_runs)
        score = min(run.get("confidence_score", 0.0) for run in query_runs)

        if fallback_used:
            label = "low"
        elif score >= 0.9:
            label = "high"
        elif score >= 0.7:
            label = "medium"
        elif score >= 0.5:
            label = "guarded"
        else:
            label = "low"

        return {
            "label": label,
            "score": round(score, 2),
            "fallback_used": fallback_used,
        }

    def _collect_expansion_terms(self, semantic_results):
        expansion_terms = []

        for entity in semantic_results.get("entities", []):
            name = entity.get("name")
            if name:
                expansion_terms.append(name)

        for relationship in semantic_results.get("relationships", []):
            for key in ("source", "target", "related_node"):
                value = relationship.get(key)
                if value:
                    expansion_terms.append(value)

        unique_terms = []
        seen_terms = set()
        for term in expansion_terms:
            normalized_term = self._normalize_query(term)
            if normalized_term and normalized_term not in seen_terms:
                seen_terms.add(normalized_term)
                unique_terms.append(term)

        return unique_terms

    def store_memory(self, MemoryBlob, build_graph: bool = True):
        total_start = time.perf_counter()
        resolve_start = time.perf_counter()
        episodic_result = self.episodic_manager.resolve_memory_for_storage(MemoryBlob)
        resolve_ms = (time.perf_counter() - resolve_start) * 1000
        print(f"[Timing] episodic_resolve_ms={resolve_ms:.1f}")

        canonical_memory_blob = episodic_result.get("memory_blob", MemoryBlob) if episodic_result else MemoryBlob
        episodic_action = episodic_result.get("action") if episodic_result else None

        if episodic_action == "duplicate":
            episodic_ms = 0.0
            print(f"[Timing] episodic_store_ms={episodic_ms:.1f}")
            print("[Timing] graph_push_ms=skipped_duplicate")
            total_ms = (time.perf_counter() - total_start) * 1000
            print(f"[Timing] store_memory_total_ms={total_ms:.1f}")
            return episodic_result

        persist_start = time.perf_counter()
        self._persist_episodic_memory(canonical_memory_blob)
        episodic_ms = (time.perf_counter() - persist_start) * 1000
        print(f"[Timing] episodic_store_ms={episodic_ms:.1f}")

        if build_graph:
            self._enqueue_graph_memory(canonical_memory_blob)
            print("[Timing] graph_push_ms=queued")
        else:
            print("[Timing] graph_push_ms=skipped")

        total_ms = (time.perf_counter() - total_start) * 1000
        print(f"[Timing] store_memory_total_ms={total_ms:.1f}")

    def _persist_episodic_memory(self, memory_blob):
        episodic_manager = EpisodicMemoryManager()
        episodic_manager.persist_memory(memory_blob)

    def _enqueue_graph_memory(self, memory_blob):
        type(self)._graph_queue.put(memory_blob)

    def _push_graph_memory(self, memory_blob):
        graph_manager = GraphMemoryManager()
        graph_manager.push_to_graphdb(memory_blob)
    
    def retrieve_memory_raw(self, query, depth=0, max_queries_per_hop=5):
        max_depth = max(0, int(depth))

        accumulated_episodic = {}
        accumulated_entities = {}
        accumulated_relationships = {}
        accumulated_memory_ids = set()
        hop_results = []

        seen_memory_ids = set()
        seen_entities = set()
        seen_relationships = set()
        seen_queries = set()

        current_queries = [query]
        seen_queries.add(self._normalize_query(query))

        for hop in range(max_depth + 1):
            next_queries = []
            hop_episodic = []
            hop_entities = []
            hop_relationships = []
            hop_query_runs = []
            hop_queries = list(current_queries)

            for seed_query in current_queries:
                rewritten_query = self._rewrite_query_for_retrieval(seed_query)
                query_embedding = get_embedding(rewritten_query)
                episodic_results = []
                used_threshold = None
                confidence = {"label": "unknown", "score": 0.0, "fallback_used": False}

                for threshold in EPISODIC_RETRIEVAL_FALLBACK_THRESHOLDS:
                    episodic_results = self.episodic_manager.retrieve_similar(
                        query_embedding,
                        threshold=threshold if threshold is not None else 0.0,
                    )
                    used_threshold = threshold
                    confidence = self._episodic_confidence_from_threshold(threshold)
                    if episodic_results:
                        break

                semantic_results = self.graph_manager.retrieve_entities_and_relationships(rewritten_query)
                hop_query_runs.append({
                    "original_query": seed_query,
                    "rewritten_query": rewritten_query,
                    "used_threshold": used_threshold,
                    "confidence_label": confidence["label"],
                    "confidence_score": confidence["score"],
                    "fallback_used": confidence["fallback_used"],
                    "episodic_count": len(episodic_results),
                    "entity_count": len(semantic_results.get("entities", [])),
                    "relationship_count": len(semantic_results.get("relationships", [])),
                })

                for memory in episodic_results:
                    memory_id = memory.get("id")
                    if not memory_id:
                        continue

                    current_score = float(memory.get("similarity", 0.0) or 0.0)
                    existing_memory = accumulated_episodic.get(memory_id)
                    if existing_memory is None or current_score > float(existing_memory.get("similarity", 0.0) or 0.0):
                        accumulated_episodic[memory_id] = memory

                    if memory_id not in seen_memory_ids:
                        seen_memory_ids.add(memory_id)
                        hop_episodic.append(memory)

                for entity in semantic_results.get("entities", []):
                    entity_name = entity.get("name")
                    entity_labels = tuple(entity.get("labels", []))
                    entity_key = (entity_name, entity_labels)
                    entity_score = float(entity.get("score", 0.0) or 0.0)
                    if entity_name:
                        existing_entity = accumulated_entities.get(entity_key)
                        if existing_entity is None or entity_score > float(existing_entity.get("score", 0.0) or 0.0):
                            accumulated_entities[entity_key] = entity

                    if entity_name and entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        hop_entities.append(entity)

                for relationship in semantic_results.get("relationships", []):
                    relationship_key = (
                        relationship.get("source"),
                        relationship.get("type"),
                        relationship.get("target"),
                        relationship.get("related_node"),
                    )
                    relationship_score = float(relationship.get("score", 0.0) or 0.0)
                    existing_relationship = accumulated_relationships.get(relationship_key)
                    if existing_relationship is None or relationship_score > float(existing_relationship.get("score", 0.0) or 0.0):
                        accumulated_relationships[relationship_key] = relationship

                    if relationship_key not in seen_relationships:
                        seen_relationships.add(relationship_key)
                        hop_relationships.append(relationship)

                if not hop_episodic and used_threshold is not None:
                    print(f"[Timing] episodic_retrieve_fallback_threshold={used_threshold}")

                if hop < max_depth:
                    expansion_terms = self._collect_expansion_terms(semantic_results)
                    for term in expansion_terms[:max_queries_per_hop]:
                        normalized_term = self._normalize_query(term)
                        if normalized_term and normalized_term not in seen_queries:
                            seen_queries.add(normalized_term)
                            next_queries.append(term)

            hop_results.append({
                "depth": hop,
                "queries": hop_queries,
                "episodic": hop_episodic,
                "entities": hop_entities,
                "relationships": hop_relationships,
                "query_runs": hop_query_runs,
                "retrieval_confidence": self._summarize_retrieval_confidence(hop_query_runs),
            })

            current_queries = next_queries
            if not current_queries:
                break

        sorted_episodic = sorted(
            accumulated_episodic.values(),
            key=lambda item: float(item.get("similarity", 0.0) or 0.0),
            reverse=True,
        )
        sorted_entities = sorted(
            accumulated_entities.values(),
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                len(item.get("relationships", [])) if isinstance(item.get("relationships", []), list) else 0,
                item.get("name", "").lower(),
            ),
            reverse=True,
        )
        sorted_relationships = sorted(
            accumulated_relationships.values(),
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                item.get("source", "") or "",
                item.get("type", "") or "",
                item.get("target", "") or "",
            ),
            reverse=True,
        )

        semantic_output = {
            "entities": sorted_entities,
            "relationships": sorted_relationships,
            "memory_ids": sorted(accumulated_memory_ids),
        }

        overall_confidence = self._summarize_retrieval_confidence(
            [hop_result.get("retrieval_confidence", {}) for hop_result in hop_results]
        )

        return {
            "episodic": sorted_episodic,
            "semantic": semantic_output,
            "entities": sorted_entities,
            "relationships": sorted_relationships,
            "memory_ids": sorted(accumulated_memory_ids),
            "depth": max_depth,
            "hop_results": hop_results,
            "retrieval_confidence": overall_confidence,
        }

    def retrieve_memory(self, query, depth=0):
        raw_results = self.retrieve_memory_raw(query, depth=depth)
        hop_results = raw_results.get("hop_results", [])
        retrieval_confidence = raw_results.get("retrieval_confidence", {})

        output_lines = []

        if retrieval_confidence:
            output_lines.append(
                f"Retrieval confidence: {retrieval_confidence.get('label', 'unknown')} "
                f"(score={retrieval_confidence.get('score', 0.0):.2f})"
            )
            if retrieval_confidence.get("fallback_used"):
                output_lines.append(
                    "Warning: an unbounded fallback was used, so some memories may be weakly relevant."
                )
            output_lines.append("")

        for hop_result in hop_results:
            depth_index = hop_result.get("depth", 0)
            hop_queries = hop_result.get("queries", [])
            hop_episodic = hop_result.get("episodic", [])
            hop_entities = hop_result.get("entities", [])
            hop_relationships = hop_result.get("relationships", [])
            hop_confidence = hop_result.get("retrieval_confidence", {})

            output_lines.append(f"Depth {depth_index}:")
            if hop_confidence:
                output_lines.append(
                    f"  Retrieval confidence: {hop_confidence.get('label', 'unknown')} "
                    f"(score={hop_confidence.get('score', 0.0):.2f})"
                )
                if hop_confidence.get("fallback_used"):
                    output_lines.append("  Warning: unbounded fallback was required for this hop.")

            if hop_queries:
                output_lines.append("  Queries:")
                for item in hop_queries:
                    output_lines.append(f"  - {item}")

            if hop_episodic:
                output_lines.append("  Relevant episodic memories:")
                for memory in hop_episodic:
                    content = memory.get("content")
                    if content:
                        output_lines.append(f"  - {content}")
            else:
                output_lines.append("  Relevant episodic memories: None found")

            if hop_relationships:
                output_lines.append("  Relevant graph facts:")
                for relationship in hop_relationships:
                    source = relationship.get("source")
                    relation_type = relationship.get("type")
                    target = relationship.get("target")
                    if source and relation_type and target:
                        output_lines.append(f"  - {source} -> {relation_type} -> {target}")
            else:
                output_lines.append("  Relevant graph facts: None found")

            if hop_entities:
                output_lines.append("  Query entities:")
                for entity in hop_entities:
                    name = entity.get("name")
                    labels = entity.get("labels", [])
                    if name:
                        if labels:
                            output_lines.append(f"  - {name} ({', '.join(labels)})")
                        else:
                            output_lines.append(f"  - {name}")
            else:
                output_lines.append("  Query entities: None found")

            output_lines.append("")

        return "\n".join(line for line in output_lines if line is not None).rstrip()
