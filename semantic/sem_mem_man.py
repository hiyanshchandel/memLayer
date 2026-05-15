import json
from memory_blob.definition import MemoryBlob
from config import GRAPH_CONTEXT_ENTITY_LIMIT, GRAPH_PUSH_MAX_RETRIES, GRAPH_PUSH_RETRY_BASE_SECONDS, neo4j_extraction_prompt, semantic_extraction_model
from clients.graphdb_client import graphdb_client
from clients.openai_client import openai_client
import re
import time

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "around", "as", "ask", "asked", "asking", "at", "be", "did", "do",
    "does", "doing", "for", "from", "give", "go", "goes", "got", "had", "has", "have", "how", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "tell", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "whose", "why", "with", "would", "you", "your"
}


class GraphMemoryManager:
    def __init__(self):
        self.client = graphdb_client
        self.openai_client = openai_client

    def _format_known_entities_context(self, known_entities):
        if not known_entities:
            return ""

        lines = ["Known entities already seen in this ingest session:"]
        for entity in known_entities[:GRAPH_CONTEXT_ENTITY_LIMIT]:
            name = entity.get("name")
            label = entity.get("label")
            count = entity.get("count", 0)
            if name and label:
                lines.append(f"- {name} [{label}] (seen {count} times)")

        return "\n".join(lines)

    def _normalize_name_key(self, value):
        if not value:
            return ""

        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _build_query_name_variants(self, entity_names):
        variants = []
        seen = set()

        for name in entity_names:
            normalized_name = self._normalize_name_key(name)
            if not normalized_name or normalized_name in seen:
                continue

            seen.add(normalized_name)
            variants.append({
                "raw": name,
                "normalized": normalized_name,
                "tokens": [token for token in re.split(r"\s+", str(name).strip()) if token],
            })

        return variants

    def _score_query_name_match(self, node_name, query_variant, match_type):
        node_key = self._normalize_name_key(node_name)
        variant_key = query_variant.get("normalized", "")
        node_tokens = {token for token in re.findall(r"[a-z0-9']+", self._strip_honorifics(node_name).lower()) if token}
        variant_tokens = set(query_variant.get("tokens", []))

        token_overlap = len(node_tokens & variant_tokens)
        token_union = len(node_tokens | variant_tokens) or 1
        overlap_score = token_overlap / token_union

        containment_score = 0.0
        if node_key and variant_key and (node_key == variant_key):
            containment_score = 1.0
        elif node_key and variant_key and (node_key in variant_key or variant_key in node_key):
            containment_score = 0.9

        score = max(overlap_score, containment_score)
        if match_type == "exact":
            score = max(score, 1.0)

        return round(min(score, 1.0), 3)

    def _relationship_key(self, relationship):
        return (
            relationship.get("source"),
            relationship.get("type"),
            relationship.get("target"),
            relationship.get("related_node"),
            relationship.get("direction"),
        )

    def _extract_query_name_candidates(self, query):
        if not query:
            return []

        raw_tokens = re.findall(r"[a-z0-9']+", str(query).lower())
        content_tokens = [token for token in raw_tokens if token not in QUERY_STOPWORDS]

        candidates = []
        seen = set()

        def add_candidate(value):
            normalized_value = self._normalize_name_key(value)
            if not normalized_value or normalized_value in seen:
                return

            seen.add(normalized_value)
            candidates.append(value.strip())

        if content_tokens:
            add_candidate(" ".join(content_tokens))

        for token in content_tokens:
            add_candidate(token)

        for index in range(len(content_tokens) - 1):
            add_candidate(" ".join(content_tokens[index:index + 2]))

        for index in range(len(content_tokens) - 2):
            add_candidate(" ".join(content_tokens[index:index + 3]))

        return candidates

    def _strip_honorifics(self, value):
        if not value:
            return ""

        text = str(value)
        text = re.sub(r"(?i)\b(mr|mrs|ms|miss|dr|prof|professor|sir|madam|lady|lord)\.?(?=\s)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _token_set(self, value):
        cleaned = self._strip_honorifics(value).lower()
        cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
        tokens = [token for token in cleaned.split() if token]
        return set(tokens)

    def _canonicalize_name(self, name, known_entities=None):
        if not name:
            return None

        cleaned_name = self._strip_honorifics(name)
        name_key = self._normalize_name_key(cleaned_name)
        if not name_key:
            return cleaned_name or name

        best_match = None
        best_score = 0
        name_tokens = self._token_set(cleaned_name)

        for entity in known_entities or []:
            known_name = entity.get("name")
            if not known_name:
                continue

            known_cleaned = self._strip_honorifics(known_name)
            known_key = self._normalize_name_key(known_cleaned)
            if not known_key:
                continue

            if name_key == known_key:
                return known_cleaned

            known_tokens = self._token_set(known_cleaned)
            if not name_tokens or not known_tokens:
                continue

            token_overlap = len(name_tokens & known_tokens)
            token_union = len(name_tokens | known_tokens) or 1
            overlap_score = token_overlap / token_union

            substring_match = name_key in known_key or known_key in name_key
            if substring_match:
                overlap_score = max(overlap_score, 0.85)

            if overlap_score > best_score:
                best_score = overlap_score
                best_match = known_cleaned

        if best_match and best_score >= 0.5:
            return best_match

        return cleaned_name

    def _normalize_extracted_data(self, data):
        normalized_entities = []
        id_map = {}
        seen_entity_names = {}

        for entity in data.get("entities", []):
            entity_id = str(entity.get("id", "")).strip()
            entity_label = str(entity.get("label", "")).strip()
            entity_name = str(entity.get("name", "")).strip()

            if not entity_id or not entity_label or not entity_name:
                continue

            entity_key = self._normalize_name_key(entity_name)
            if entity_key in seen_entity_names:
                canonical_entity = seen_entity_names[entity_key]
                id_map[entity_id] = canonical_entity
                continue

            normalized_entities.append({
                "id": entity_id,
                "label": entity_label,
                "name": entity_name,
            })
            canonical_entity = normalized_entities[-1]
            seen_entity_names[entity_key] = canonical_entity
            id_map[entity_id] = canonical_entity

        normalized_relationships = []
        seen_relationships = set()

        for relationship in data.get("relationships", []):
            start_id = str(relationship.get("start_id", "")).strip()
            end_id = str(relationship.get("end_id", "")).strip()
            relation_type = str(relationship.get("type", "")).strip()

            if not start_id or not end_id or not relation_type:
                continue
            if start_id == end_id:
                continue
            if start_id not in id_map or end_id not in id_map:
                continue

            relationship_key = (start_id, end_id, relation_type)
            if relationship_key in seen_relationships:
                continue

            seen_relationships.add(relationship_key)
            normalized_relationships.append({
                "start_id": start_id,
                "end_id": end_id,
                "type": relation_type,
            })

        return {
            "entities": normalized_entities,
            "relationships": normalized_relationships,
        }

    def extract_entities_and_relationships(self, role = neo4j_extraction_prompt, Memory = MemoryBlob, known_entities=None) -> dict:
        start = time.perf_counter()
        known_entities_context = self._format_known_entities_context(known_entities)

        for attempt in range(GRAPH_PUSH_MAX_RETRIES):
            try:
                user_content = Memory.content
                if known_entities_context:
                    user_content = f"{known_entities_context}\n\nCurrent text:\n{Memory.content}"

                response = self.openai_client.chat.completions.create(
                    model=semantic_extraction_model,
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": user_content}
                    ]
                )
                ans = response.choices[0].message.content
                clean_json_str = re.sub(r"^```json\s*|\s*```$", "", ans.strip())
                data = json.loads(clean_json_str)
                data = self._normalize_extracted_data(data)
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"[Timing] graph_llm_extract_ms={elapsed_ms:.1f}")
                return data
            except Exception as exc:
                error_text = str(exc)
                is_rate_limit = "rate_limit_exceeded" in error_text or "Error code: 429" in error_text

                if not is_rate_limit or attempt >= GRAPH_PUSH_MAX_RETRIES - 1:
                    raise

                delay_seconds = GRAPH_PUSH_RETRY_BASE_SECONDS * (2 ** attempt)
                print(f"[Timing] graph_llm_rate_limit_retry={attempt + 1}, delay_seconds={delay_seconds:.2f}")
                time.sleep(delay_seconds)

        raise RuntimeError("Graph extraction failed after retries.")
    
    def retrieve_entities_and_relationships(self, query: str):
        query_memory = type("QueryMemory", (), {"content": query})()
        extracted_entities = []
        try:
            extracted = self.extract_entities_and_relationships(Memory=query_memory)
            extracted_entities = extracted.get("entities", [])
        except Exception:
            extracted_entities = []

        entity_names = self._extract_query_name_candidates(query)
        entity_names.extend(entity.get("name") for entity in extracted_entities if entity.get("name"))
        entity_names = list(dict.fromkeys(name for name in entity_names if name))
        if not entity_names:
            return {
                "entities": [],
                "relationships": [],
                "memory_ids": []
            }

        query_variants = self._build_query_name_variants(entity_names)
        if not query_variants:
            return {
                "entities": [],
                "relationships": [],
                "memory_ids": []
            }

        with self.client.session() as session:
            exact_result = session.run(
                """
                UNWIND $entity_names AS entity_name
                MATCH (n {name: entity_name})
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN
                    n AS node,
                    entity_name AS matched_query,
                    collect(DISTINCT {
                        type: type(r),
                        direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
                        source: startNode(r).name,
                        target: endNode(r).name,
                        related_node: m.name
                    }) AS relationships
                """,
                entity_names=entity_names,
            )

            fuzzy_result = session.run(
                """
                UNWIND $query_variants AS query_variant
                MATCH (n)
                WHERE
                    toLower(replace(coalesce(n.name, ''), '.', '')) CONTAINS query_variant.normalized
                    OR query_variant.normalized CONTAINS toLower(replace(coalesce(n.name, ''), '.', ''))
                    OR ANY(token IN query_variant.tokens WHERE
                        toLower(replace(coalesce(n.name, ''), '.', '')) CONTAINS toLower(token)
                    )
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN
                    n AS node,
                    query_variant.raw AS matched_query,
                    collect(DISTINCT {
                        type: type(r),
                        direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
                        source: startNode(r).name,
                        target: endNode(r).name,
                        related_node: m.name
                    }) AS relationships
                """,
                query_variants=query_variants,
            )

            if not entity_names:
                return {
                    "entities": [],
                    "relationships": [],
                    "memory_ids": []
                }

            entities = []
            relationships = {}
            memory_ids = set()
            seen_entity_keys = set()
            query_variant_lookup = {variant.get("raw"): variant for variant in query_variants}

            def add_records(result, match_type):
                for record in result:
                    node = record["node"]
                    node_name = node.get("name")
                    node_key = self._normalize_name_key(node_name)
                    if not node_name or node_key in seen_entity_keys:
                        continue

                    seen_entity_keys.add(node_key)

                    node_data = dict(node)
                    node_data["labels"] = list(node.labels)
                    node_data["match_type"] = match_type
                    matched_query = record.get("matched_query")
                    node_data["matched_query"] = matched_query
                    query_variant = query_variant_lookup.get(matched_query, {
                        "normalized": self._normalize_name_key(matched_query),
                        "tokens": [token for token in re.findall(r"[a-z0-9']+", str(matched_query).lower()) if token],
                    })
                    relationship_count = len(record.get("relationships", []))
                    node_data["score"] = self._score_query_name_match(
                        node_name,
                        query_variant,
                        match_type,
                    )
                    if relationship_count:
                        node_data["score"] = round(min(1.0, node_data["score"] + min(0.1, relationship_count / 50.0)), 3)
                    entities.append(node_data)

                    node_memory_id = node.get("memory_id")
                    if node_memory_id:
                        memory_ids.add(node_memory_id)

                    node_memory_ids = node.get("memory_ids")
                    if node_memory_ids:
                        for memory_id in node_memory_ids:
                            memory_ids.add(memory_id)

                    for relationship in record["relationships"]:
                        if relationship and relationship.get("type"):
                            relationship_data = dict(relationship)
                            relationship_data["match_type"] = match_type
                            relationship_data["matched_query"] = record.get("matched_query")
                            relationship_data["score"] = node_data["score"]
                            relationship_key = self._relationship_key(relationship_data)
                            existing_relationship = relationships.get(relationship_key)
                            if existing_relationship is None or relationship_data["score"] > existing_relationship.get("score", 0.0):
                                relationships[relationship_key] = relationship_data

            add_records(exact_result, "exact")
            add_records(fuzzy_result, "fuzzy")

            entities.sort(
                key=lambda item: (
                    float(item.get("score", 0.0) or 0.0),
                    1 if item.get("match_type") == "exact" else 0,
                    item.get("name", "").lower(),
                ),
                reverse=True,
            )

            ranked_relationships = sorted(
                relationships.values(),
                key=lambda item: (
                    float(item.get("score", 0.0) or 0.0),
                    1 if item.get("match_type") == "exact" else 0,
                    item.get("source", "") or "",
                    item.get("type", "") or "",
                    item.get("target", "") or "",
                ),
                reverse=True,
            )

            return {
                "entities": entities,
                "relationships": ranked_relationships,
                "memory_ids": list(memory_ids),
            }

    def push_to_graphdb(self, Memory = MemoryBlob, known_entities=None): 
        push_start = time.perf_counter()
        data = self.extract_entities_and_relationships(role = neo4j_extraction_prompt, Memory = Memory, known_entities=known_entities)
        entity_lookup = {e["id"]: e for e in data["entities"]}

        with self.client.session() as session:
            # Create all nodes
            for entity in data["entities"]:
                memory_id = getattr(Memory, "id", None)
                session.run(
                    f"""
                    MERGE (n:{entity['label']} {{name: $name}})
                    SET n.memory_id = coalesce(n.memory_id, $memory_id),
                        n.memory_ids = CASE
                            WHEN n.memory_ids IS NULL THEN [$memory_id]
                            WHEN $memory_id IN n.memory_ids THEN n.memory_ids
                            ELSE n.memory_ids + $memory_id
                        END
                    """,
                    name=entity["name"],
                    memory_id=memory_id,
                )

            # Create relationships
            for relation in data["relationships"]:
                start_entity = entity_lookup.get(relation["start_id"])
                end_entity = entity_lookup.get(relation["end_id"])
                rel_type = relation["type"]

                if not start_entity or not end_entity:
                    continue

                session.run(
                    f"""
                    MATCH (a:{start_entity['label']} {{name: $start_name}}),
                        (b:{end_entity['label']} {{name: $end_name}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """,
                    start_name=start_entity["name"],
                    end_name=end_entity["name"]
                )
        elapsed_ms = (time.perf_counter() - push_start) * 1000
        print(f"[Timing] graph_push_total_ms={elapsed_ms:.1f}")
        return data
        





    
