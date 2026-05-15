
import importlib
import hashlib
import html
import io
import math
import os
import re
import sqlite3
import threading
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from qdrant_client.models import PointIdsList

from clients.graphdb_client import graphdb_client
from clients.openai_client import openai_client
from clients.vector_client import vec_client
from config import EPISODIC_COLLECTION_NAME, EPISODIC_MEMORY_DB
from embeddings import get_embeddings
from memory_agent.mem_agent import MemoryAgent
from memory_blob.definition import MemoryBlob

try:
    PdfReader = importlib.import_module("pypdf").PdfReader
except Exception:
    PdfReader = None


LABEL_COLORS = {
    "Person": "#7dd3fc",
    "Project": "#f59e0b",
    "Technology": "#34d399",
    "Algorithm": "#f97316",
    "Organization": "#a78bfa",
    "Location": "#fb7185",
    "Concept": "#60a5fa",
    "Event": "#f472b6",
    "Document": "#22c55e",
    "Attribute": "#eab308",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-main: #fdfdfd;
            --card-bg: #ffffff;
            --border: #e4e4e7;
            --text-primary: #09090b;
            --text-muted: #71717a;
            --accent: #18181b;
            --accent-hover: #27272a;
            --primary: #3b82f6;
            --radius: 12px;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            background-image: none !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        h1, h2, h3, h4, h5, h6, p, span, div, label {
            font-family: 'Inter', sans-serif !important;
        }

        .hero-shell {
            padding: 2.5rem 0;
            text-align: center;
            margin-bottom: 1.5rem;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .hero-title {
            font-size: 2.75rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 1rem;
            letter-spacing: -0.03em;
        }
        .hero-copy {
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 640px;
            margin: 0 auto 1.5rem auto;
            line-height: 1.6;
        }
        .hero-shell::after { display: none !important; }

        .chip-row {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .chip {
            padding: 0.35rem 0.8rem;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 999px;
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
            box-shadow: var(--shadow-sm);
        }
        .chip strong {
            color: var(--text-primary);
            font-weight: 600;
        }

        .panel-shell, .summary-shell, .query-shell, .reset-shell,
        .input-card, .settings-card, .status-card, .metric-card, .reset-card, .graph-empty, .graph-shell, .query-box {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow-sm) !important;
            margin-bottom: 1rem;
        }

        .panel-title, .summary-title, .query-title, .reset-title, .graph-title, .section-eyebrow, .mini-eyebrow, h4 {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0 !important;
            letter-spacing: -0.01em !important;
            text-transform: none !important;
        }
        .panel-copy, .summary-copy, .query-copy, .reset-copy, .graph-copy, p, .footer-note, .progress-copy {
            color: var(--text-muted) !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
            margin-top: 0 !important;
            margin-bottom: 0.75rem !important;
        }

        .stButton > button {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            box-shadow: var(--shadow-sm) !important;
            height: auto !important;
        }
        .stButton > button:hover {
            background: var(--accent-hover) !important;
            transform: translateY(-1px);
        }

        .stTextInput input, .stTextArea textarea {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 0.6rem 0.8rem !important;
            color: var(--text-primary) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }

        div[data-testid="stWidgetLabel"] label, div[data-testid="stWidgetLabel"] p {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-main) !important;
            border: 1px dashed var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 2rem !important;
            transition: border-color 0.2s ease !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--primary) !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--text-muted) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: white !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 1px solid var(--border) !important;
            gap: 1.5rem !important;
            padding: 0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 0.25rem !important;
            color: var(--text-muted) !important;
            border: none !important;
            background: transparent !important;
            font-weight: 500 !important;
            border-radius: 0 !important;
            min-height: auto !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
            border-bottom: 2px solid var(--text-primary) !important;
            box-shadow: none !important;
        }

        .metric-grid, .query-results, .reset-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .query-box {
            background: #f8fafc !important;
            border-color: #d4d4d8 !important;
        }

        .query-output {
            margin-top: 0.5rem;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            background: #f8fafc !important;
            border: 1px solid #d4d4d8;
            color: #111827 !important;
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Monaco, Consolas, monospace;
            font-size: 0.92rem;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-x: auto;
        }

        .query-output pre {
            margin: 0;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        .chat-shell {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow-sm) !important;
            margin-bottom: 1rem;
        }

        .chat-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }

        .chat-transcript {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .chat-empty {
            padding: 1rem;
            border: 1px dashed var(--border);
            border-radius: var(--radius);
            background: #fafafa;
            color: var(--text-muted);
            line-height: 1.6;
        }

        .chat-meta {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 0.35rem;
        }

        .chat-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #fafafa;
            color: var(--text-muted);
            font-size: 0.8rem;
            font-weight: 500;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff, #fafafa);
            padding: 0.35rem 0.4rem;
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: var(--text-primary) !important;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }
        .metric-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .metric-detail {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .graph-shell::before { display: none !important; }
        .graph-header { margin-bottom: 1.5rem; }
        .graph-title { font-size: 1.25rem !important; }
        .graph-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }
        .graph-chip {
            padding: 0.25rem 0.6rem;
            background: var(--bg-main);
            border: 1px solid var(--border);
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .stRadio [role="radiogroup"] label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }
        .stSlider [data-baseweb="slider"] label {
            color: var(--text-primary) !important;
        }

        @media (max-width: 768px) {
            .hero-title { font-size: 2rem; }
            .metric-grid, .query-results, .reset-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def split_text(text: str, chunk_size: int, overlap: int):
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            split_point = text.rfind("\n", start, end)
            if split_point == -1:
                split_point = text.rfind(". ", start, end)
            if split_point != -1 and split_point > start:
                end = split_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_pdf_text(uploaded_file):
    if PdfReader is None:
        raise RuntimeError("PDF support is unavailable because 'pypdf' is not installed.")

    pdf_bytes = io.BytesIO(uploaded_file.read())
    reader = PdfReader(pdf_bytes)
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append((index, page_text))

    return pages


def extract_input_text(uploaded_file, manual_text: str):
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".txt"):
            return "txt", uploaded_file.read().decode("utf-8", errors="ignore"), uploaded_file.name
        if file_name.endswith(".pdf"):
            return "pdf", extract_pdf_text(uploaded_file), uploaded_file.name
        raise ValueError("Only .txt and .pdf files are supported.")

    return "text", manual_text, "manual_input"


def build_memory_blobs(chunk_records, memory_type: str, batch_size: int = 16):
    memory_blobs = []

    for batch_start in range(0, len(chunk_records), batch_size):
        batch_records = chunk_records[batch_start: batch_start + batch_size]
        batch_texts = [record[0] for record in batch_records]
        batch_embeddings = get_embeddings(batch_texts)

        for (chunk_text, tags), embedding in zip(batch_records, batch_embeddings):
            memory_blobs.append(
                MemoryBlob(
                    content=chunk_text,
                    memory_type=memory_type,
                    embedding=embedding,
                    tags=tags,
                )
            )

    return memory_blobs


def prepare_chunk_records(source_kind: str, extracted, source_name: str, content_kind: str):
    chunk_records = []

    if source_kind == "pdf":
        for page_number, page_text in extracted:
            page_chunks = split_text(page_text, st.session_state["chunk_size"], st.session_state["overlap"])
            for page_chunk_index, chunk_text in enumerate(page_chunks, start=1):
                chunk_records.append(
                    (
                        chunk_text,
                        {
                            "source_name": source_name,
                            "source_type": source_kind,
                            "content_kind": content_kind,
                            "page_number": page_number,
                            "page_chunk_index": page_chunk_index,
                            "chunk_index": len(chunk_records) + 1,
                        },
                    )
                )
    else:
        chunks = split_text(extracted, st.session_state["chunk_size"], st.session_state["overlap"])
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_records.append(
                (
                    chunk_text,
                    {
                        "source_name": source_name,
                        "source_type": source_kind,
                        "chunk_index": chunk_index,
                        "content_kind": content_kind,
                    },
                )
            )

    return chunk_records


def clear_sqlite_memory():
    conn = sqlite3.connect(EPISODIC_MEMORY_DB)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories")
        conn.commit()
    finally:
        conn.close()


def clear_qdrant_memory():
    scroll_result = vec_client.scroll(
        collection_name=EPISODIC_COLLECTION_NAME,
        limit=100,
        with_payload=False,
        with_vectors=False,
    )

    while True:
        points, next_offset = scroll_result
        if not points:
            break

        point_ids = [point.id for point in points]
        vec_client.delete(
            collection_name=EPISODIC_COLLECTION_NAME,
            points_selector=PointIdsList(points=point_ids),
        )

        if next_offset is None:
            break

        scroll_result = vec_client.scroll(
            collection_name=EPISODIC_COLLECTION_NAME,
            limit=100,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )


def clear_graph_memory():
    with graphdb_client.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def clear_all_memory():
    clear_graph_memory()
    clear_sqlite_memory()
    clear_qdrant_memory()


def answer_query_from_retrieval(user_query: str, retrieval_output: str) -> str:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = [
        {
            "role": "system",
            "content": (
                "You answer the user only using the retrieved memory output. "
                "If the retrieved output does not contain enough information, say so plainly. "
                "Keep the answer concise and do not invent facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{user_query}\n\n"
                f"Retrieved memory output:\n{retrieval_output}\n\n"
                "Answer the user based on the retrieved memory output."
            ),
        },
    ]

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=prompt,
    )
    return (response.choices[0].message.content or "").strip()


def format_chat_memory_context(raw_results: dict[str, Any], max_items: int = 4) -> str:
    confidence = raw_results.get("retrieval_confidence", {}) or {}
    episodic = raw_results.get("episodic", []) or []
    entities = raw_results.get("entities", []) or []
    relationships = raw_results.get("relationships", []) or []

    lines = []
    if confidence:
        score = float(confidence.get("score", 0.0) or 0.0)
        lines.append(f"Retrieval confidence: {confidence.get('label', 'unknown')} (score={score:.2f})")
        if confidence.get("fallback_used"):
            lines.append("Warning: fallback retrieval widened the search, so some memories may be weakly relevant.")

    if episodic:
        lines.append("Relevant episodic memories:")
        for memory in episodic[:max_items]:
            content = (memory.get("content") or "").strip()
            if content:
                lines.append(f"- {content}")

    if entities:
        lines.append("Relevant graph entities:")
        for entity in entities[:max_items]:
            name = entity.get("name")
            label = entity.get("label") or ", ".join(entity.get("labels", []))
            if name:
                suffix = f" [{label}]" if label else ""
                lines.append(f"- {name}{suffix}")

    if relationships:
        lines.append("Relevant graph relationships:")
        for relationship in relationships[:max_items]:
            source = relationship.get("source")
            relation_type = relationship.get("type")
            target = relationship.get("target")
            if source and relation_type and target:
                lines.append(f"- {source} -> {relation_type} -> {target}")

    return "\n".join(lines).strip() or "No relevant memories found."


def build_chat_prompt_history(chat_messages: list[dict[str, str]], limit: int = 8):
    prompt_messages = []
    for message in chat_messages[-limit:]:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "assistant"} and content:
            prompt_messages.append({"role": role, "content": content})
    return prompt_messages


def answer_chat_with_memory(user_input: str, memory_context: str, chat_messages: list[dict[str, str]], retrieval_confidence: dict[str, Any]) -> str:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    confidence_label = retrieval_confidence.get("label", "unknown") if retrieval_confidence else "unknown"
    confidence_score = float(retrieval_confidence.get("score", 0.0) or 0.0) if retrieval_confidence else 0.0
    fallback_used = bool(retrieval_confidence.get("fallback_used")) if retrieval_confidence else False

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful chat assistant for MemLayer. Answer naturally and concisely. "
                "Use the memory context when it is relevant, but do not invent facts. "
                "If the memory context looks weak or uncertain, say so briefly instead of overcommitting."
            ),
        },
        {
            "role": "system",
            "content": (
                f"Memory context confidence: {confidence_label} (score={confidence_score:.2f}, fallback_used={fallback_used})\n\n"
                f"Memory context:\n{memory_context}"
            ),
        },
    ]

    prompt.extend(build_chat_prompt_history(chat_messages))
    prompt.append({"role": "user", "content": user_input})

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=prompt,
    )
    return (response.choices[0].message.content or "").strip()


def queue_chat_memory_ingest(user_input: str, assistant_output: str, retrieval_meta: dict[str, Any]):
    def worker():
        try:
            memory_agent = MemoryAgent()
            confidence = retrieval_meta.get("retrieval_confidence", {}) or {}
            memory_blob = MemoryBlob(
                content=(
                    f"Conversation turn\nUser: {user_input}\nAssistant: {assistant_output}"
                ),
                memory_type="conversation_memory",
                tags={
                    "source_type": "chat",
                    "content_kind": "conversation_turn",
                    "retrieval_confidence": confidence.get("label", "unknown"),
                    "retrieval_score": confidence.get("score", 0.0),
                    "fallback_used": confidence.get("fallback_used", False),
                },
            )
            memory_agent.store_memory(memory_blob, build_graph=True)
        except Exception as exc:
            print(f"[Timing] chat_memory_ingest_error={exc}")

    threading.Thread(target=worker, daemon=True).start()


def create_graph_state():
    return {
        "entities": {},
        "relationships": set(),
        "active_entities": set(),
        "events": [],
        "last_stage": "Waiting for ingest",
    }


def shorten_text(value: str, limit: int = 18) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def update_graph_state(graph_state: dict[str, Any], graph_data: dict[str, Any], chunk_index: int, total_chunks: int):
    entity_lookup = {entity["id"]: entity for entity in graph_data.get("entities", [])}
    active_entities = set()

    for entity in graph_data.get("entities", []):
        entity_key = f"{entity['label']}::{entity['name']}"
        record = graph_state["entities"].setdefault(
            entity_key,
            {
                "name": entity["name"],
                "label": entity["label"],
                "count": 0,
                "last_seen": 0,
            },
        )
        record["name"] = entity["name"]
        record["label"] = entity["label"]
        record["count"] += 1
        record["last_seen"] = chunk_index
        active_entities.add(entity["name"])

    for relation in graph_data.get("relationships", []):
        start_entity = entity_lookup.get(relation.get("start_id"))
        end_entity = entity_lookup.get(relation.get("end_id"))
        if not start_entity or not end_entity:
            continue
        graph_state["relationships"].add((start_entity["name"], relation.get("type", "related_to"), end_entity["name"]))

    graph_state["active_entities"] = active_entities
    graph_state["last_stage"] = f"Chunk {chunk_index}/{total_chunks} landed in GraphDB"
    graph_state["events"].append(
        {
            "chunk": chunk_index,
            "total": total_chunks,
            "status": "stored",
            "nodes": len(graph_state["entities"]),
            "edges": len(graph_state["relationships"]),
        }
    )
    graph_state["events"] = graph_state["events"][-6:]


def render_graph_html(graph_state: dict[str, Any], stage_text: str, stored_count: int, skipped_count: int, total_chunks: int, build_graph: bool) -> str:
    entities = sorted(
        graph_state.get("entities", {}).values(),
        key=lambda item: (-item["count"], item["name"].lower()),
    )
    relationships = list(graph_state.get("relationships", []))
    active_entities = set(graph_state.get("active_entities", []))

    if not build_graph:
        return f"""
        <div class="graph-empty">
            <div class="graph-eyebrow">Live graph preview</div>
            <h3>Graph building is off.</h3>
            <p>Turn it on to watch entities and relationships bloom here while chunks are being stored.</p>
            <div class="graph-empty__chips">
                <span>{stored_count} stored</span>
                <span>{skipped_count} skipped</span>
                <span>{total_chunks} chunks queued</span>
            </div>
        </div>
        """

    visible_entities = entities[:24]
    if not visible_entities:
        return f"""
        <div class="graph-empty">
            <div class="graph-eyebrow">Live graph preview</div>
            <h3>Ready for the first chunk.</h3>
            <p>As each memory lands, the graph will expand with real entities and relationships from Neo4j writes.</p>
            <div class="graph-empty__chips">
                <span>{stored_count} stored</span>
                <span>{skipped_count} skipped</span>
                <span>{total_chunks} chunks total</span>
            </div>
        </div>
        """

    width = 760
    height = 480
    center_x = 380.0
    center_y = 240.0
    node_positions: dict[str, tuple[float, float]] = {}
    node_markup = []
    edge_markup = []

    for index, entity in enumerate(visible_entities):
        seed_input = f"{entity['label']}::{entity['name']}"
        seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:10], 16)
        angle = (index / max(1, len(visible_entities))) * (2 * math.pi)
        radius = 128 + ((seed % 6) * 18)
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius * 0.6
        node_positions[entity["name"]] = (x, y)

        color = LABEL_COLORS.get(entity["label"], "#38bdf8")
        active_class = " active" if entity["name"] in active_entities else ""
        display_name = html.escape(shorten_text(entity["name"], 18))
        display_label = html.escape(entity["label"])
        count_text = f"{entity['count']}x"

        node_markup.append(
            f"""
            <g class="graph-node{active_class}" transform="translate({x:.1f},{y:.1f})">
                <circle class="graph-node__ring" r="21" style="stroke:{color};" />
                <circle r="12" style="fill:{color}; opacity:0.2;" />
                <text class="graph-node__label" y="38">{display_name}</text>
                <text class="graph-node__meta" y="52">{display_label} • {count_text}</text>
            </g>
            """
        )

    for start_name, rel_type, end_name in relationships[:34]:
        if start_name not in node_positions or end_name not in node_positions:
            continue
        x1, y1 = node_positions[start_name]
        x2, y2 = node_positions[end_name]
        active_link = start_name in active_entities or end_name in active_entities
        edge_class = "graph-link active" if active_link else "graph-link"
        edge_color = "#7dd3fc" if active_link else "#3b82f6"
        relation_label = html.escape(shorten_text(rel_type, 15))

        edge_markup.append(
            f"""
            <g>
                <line class="{edge_class}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{edge_color};" marker-end="url(#arrowHead)" />
                <text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2:.1f}" fill="#94a3b8" font-size="8" font-weight="600" text-anchor="middle">{relation_label}</text>
            </g>
            """
        )

    node_count = len(visible_entities)
    edge_count = len(relationships)

    return f"""
    <div class="graph-shell">
        <div class="graph-header">
            <div>
                <div class="graph-eyebrow">Live graph</div>
                <div class="graph-title">GraphDB feed is growing in real time</div>
                <div class="graph-copy">{html.escape(stage_text)}</div>
            </div>
            <div class="graph-chips">
                <span class="graph-chip">{stored_count} stored</span>
                <span class="graph-chip">{skipped_count} skipped</span>
                <span class="graph-chip">{node_count} nodes visible</span>
                <span class="graph-chip">{edge_count} edges</span>
            </div>
        </div>
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="Live knowledge graph preview">
            <defs>
                <linearGradient id="graphPulse" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#22d3ee" stop-opacity="0.85" />
                    <stop offset="55%" stop-color="#38bdf8" stop-opacity="0.38" />
                    <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.08" />
                </linearGradient>
                <marker id="arrowHead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,8 L8,4 z" fill="#7dd3fc" opacity="0.9"></path>
                </marker>
            </defs>
            <circle class="graph-core" cx="{center_x:.1f}" cy="{center_y:.1f}" r="30" />
            {''.join(edge_markup)}
            {''.join(node_markup)}
        </svg>
    </div>
    """


def render_graph_component_html(graph_state: dict[str, Any], stage_text: str, stored_count: int, skipped_count: int, total_chunks: int, build_graph: bool) -> str:
    return f"""
    <style>
    html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: 'Inter', sans-serif;
        color: #09090b;
    }}

    .graph-shell {{
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 12px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}

    .graph-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }}

    .graph-eyebrow {{
        font-size: 1.1rem;
        font-weight: 600;
        color: #09090b;
        margin-bottom: 0.5rem;
    }}

    .graph-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: #09090b;
    }}

    .graph-copy {{
        color: #71717a;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 0;
    }}

    .graph-chips {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.5rem;
    }}

    .graph-chip {{
        padding: 0.25rem 0.6rem;
        background: #fdfdfd;
        border: 1px solid #e4e4e7;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #71717a;
    }}

    .graph-shell svg {{
        width: 100%;
        height: auto;
        display: block;
        overflow: visible;
    }}

    .graph-core {{ fill: #dbeafe; opacity: 1; }}
    .graph-link {{ stroke-width: 1.8; stroke-linecap: round; opacity: 0.34; }}
    .graph-link.active {{ opacity: 1; stroke-width: 2.2; }}
    
    .graph-node {{ transform-box: fill-box; transform-origin: center; }}
    .graph-node__ring {{ fill: #ffffff; stroke-width: 2.2; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.08)); }}
    .graph-node__label {{ fill: #09090b; font-size: 11px; font-weight: 600; text-anchor: middle; }}
    .graph-node__meta {{ fill: #71717a; font-size: 9px; font-weight: 500; text-anchor: middle; }}

    .graph-empty {{
        min-height: 365px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.5rem;
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 12px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
    }}

    .graph-empty h3 {{ margin: 0; color: #09090b; font-size: 1.25rem; font-weight: 600; }}
    .graph-empty p {{ margin: 0; color: #71717a; line-height: 1.6; font-size: 0.95rem; }}

    .graph-empty__chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }}
    .graph-empty__chips span {{
        padding: 0.25rem 0.6rem;
        background: #fdfdfd;
        border: 1px solid #e4e4e7;
        border-radius: 999px;
        color: #71717a;
        font-size: 0.8rem;
        font-weight: 500;
    }}
    </style>
    {render_graph_html(graph_state, stage_text, stored_count, skipped_count, total_chunks, build_graph)}
    """


def render_graph_panel(target, graph_state: dict[str, Any], stage_text: str, stored_count: int, skipped_count: int, total_chunks: int, build_graph: bool):
    target.empty()
    with target.container():
        components.html(
            render_graph_component_html(
                graph_state,
                stage_text=stage_text,
                stored_count=stored_count,
                skipped_count=skipped_count,
                total_chunks=total_chunks,
                build_graph=build_graph,
            ),
            height=840,
            scrolling=False,
        )


def render_summary_cards(stored_count: int, skipped_count: int, total_chunks: int, node_count: int, edge_count: int):
    cards = [
        ("Chunks stored", stored_count, f"{total_chunks} chunks processed"),
        ("Duplicates skipped", skipped_count, "Exact duplicate memories were dropped"),
        ("Graph nodes", node_count, "Visible in the current live preview"),
        ("Graph edges", edge_count, "Relationships written during this ingest"),
    ]
    cols = st.columns(4)
    for column, (label, value, detail) in zip(cols, cards):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{html.escape(label)}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-detail">{html.escape(detail)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_live_ingest(memory_agent: MemoryAgent, memory_blobs, build_graph: bool, graph_placeholder, status_placeholder, progress_placeholder):
    total_chunks = len(memory_blobs)
    graph_state = create_graph_state()
    stored_count = 0
    skipped_count = 0

    if total_chunks == 0:
        status_placeholder.markdown(
            "<div class='status-card'><div class='mini-eyebrow'>No chunks</div><p>Nothing to store yet.</p></div>",
            unsafe_allow_html=True,
        )
        return graph_state, stored_count, skipped_count

    for index, memory_blob in enumerate(memory_blobs, start=1):
        status_placeholder.markdown(
            f"""
            <div class="status-card">
                <div class="mini-eyebrow">Ingesting</div>
                <div class="panel-title" style="margin:0 0 0.2rem 0;">Chunk {index} of {total_chunks}</div>
                <div class="panel-copy" style="margin:0;">Building embeddings, resolving duplicates, then writing to memory stores.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        resolution = memory_agent.episodic_manager.resolve_memory_for_storage(memory_blob)
        canonical_blob = resolution.get("memory_blob", memory_blob)

        if resolution["action"] == "duplicate":
            skipped_count += 1
            graph_state["events"].append(
                {
                    "chunk": index,
                    "total": total_chunks,
                    "status": "duplicate",
                    "nodes": len(graph_state["entities"]),
                    "edges": len(graph_state["relationships"]),
                }
            )
            graph_state["events"] = graph_state["events"][-6:]
            progress_placeholder.progress(int((index / total_chunks) * 100))
            render_graph_panel(
                graph_placeholder,
                graph_state=graph_state,
                stage_text=f"Chunk {index}/{total_chunks} skipped as duplicate",
                stored_count=stored_count,
                skipped_count=skipped_count,
                total_chunks=total_chunks,
                build_graph=build_graph,
            )
            continue

        memory_agent.episodic_manager.persist_memory(canonical_blob)
        stored_count += 1

        if build_graph:
            context_key = MemoryAgent._graph_context_key(canonical_blob)
            known_entities = MemoryAgent._select_known_entities(context_key)
            graph_data = memory_agent.graph_manager.push_to_graphdb(canonical_blob, known_entities=known_entities)
            MemoryAgent._update_graph_context(context_key, graph_data)
            update_graph_state(graph_state, graph_data, index, total_chunks)
            graph_state["events"].append(
                {
                    "chunk": index,
                    "total": total_chunks,
                    "status": "graph",
                    "nodes": len(graph_state["entities"]),
                    "edges": len(graph_state["relationships"]),
                }
            )
            graph_state["events"] = graph_state["events"][-6:]
            stage_text = graph_state["last_stage"]
        else:
            graph_state["events"].append(
                {
                    "chunk": index,
                    "total": total_chunks,
                    "status": "stored",
                    "nodes": len(graph_state["entities"]),
                    "edges": len(graph_state["relationships"]),
                }
            )
            graph_state["events"] = graph_state["events"][-6:]
            stage_text = f"Chunk {index}/{total_chunks} stored in episodic memory"

        progress_placeholder.progress(int((index / total_chunks) * 100))
        render_graph_panel(
            graph_placeholder,
            graph_state=graph_state,
            stage_text=stage_text,
            stored_count=stored_count,
            skipped_count=skipped_count,
            total_chunks=total_chunks,
            build_graph=build_graph,
        )

    return graph_state, stored_count, skipped_count


def render_top_hero():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-eyebrow">MemLayer ingest studio</div>
            <h1 class="hero-title">Turn loose text into a live memory graph.</h1>
            <div class="hero-copy">
                Upload a PDF or TXT file, paste raw notes, and watch each chunk land in memory with a polished live feed.
                The preview below grows from real graph writes so you can see the shape of the knowledge graph while ingest is still running.
            </div>
            <div class="chip-row">
                <span class="chip"><strong>SQLite</strong> episodic memory</span>
                <span class="chip"><strong>Qdrant</strong> vector storage</span>
                <span class="chip"><strong>Neo4j</strong> semantic graph</span>
                <span class="chip"><strong>Live</strong> chunk progress</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ingest_tab():
    left_col, right_col = st.columns([0.76, 1.24], gap="medium")
    graph_placeholder = None
    status_placeholder = None
    progress_placeholder = None

    with right_col:
        summary_shell = st.container()
        with summary_shell:
            st.markdown(
                """
                <div class="summary-shell">
                    <div class="summary-title">Live graph preview</div>
                    <div class="summary-copy">The graph below updates chunk-by-chunk while memories are written.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        progress_placeholder = st.progress(0)
        status_placeholder = st.empty()
        graph_placeholder = st.empty()

        last_summary = st.session_state.get("last_ingest_summary")
        last_graph_state = st.session_state.get("last_graph_state")
        if last_summary and last_graph_state and not st.session_state.get("ingest_running"):
            render_summary_cards(
                last_summary["stored_count"],
                last_summary["skipped_count"],
                last_summary["total_chunks"],
                last_summary["node_count"],
                last_summary["edge_count"],
            )
            render_graph_panel(
                graph_placeholder,
                graph_state=last_graph_state,
                stage_text=last_summary.get("stage_text", "Latest stored graph"),
                stored_count=last_summary["stored_count"],
                skipped_count=last_summary["skipped_count"],
                total_chunks=last_summary["total_chunks"],
                build_graph=last_summary["build_graph"],
            )
        else:
            render_summary_cards(0, 0, 0, 0, 0)
            render_graph_panel(graph_placeholder, create_graph_state(), "Waiting for your first ingest", 0, 0, 0, True)

    with left_col:
        st.markdown(
            """
            <div class="panel-shell">
                <div class="panel-title">Ingest source</div>
                <div class="panel-copy">Load text directly or upload a PDF/TXT file. You can keep the graph off for a faster ingest pass, or turn it on to stream real graph writes into the preview.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.setdefault("chunk_size", 1200)
        st.session_state.setdefault("overlap", 150)

        settings_grid = st.container()
        with settings_grid:
            st.markdown(
                """
                <div class="settings-card">
                    <div class="section-eyebrow">Ingest settings</div>
                    <div class="settings-grid">
                        <div class="wide">
                """,
                unsafe_allow_html=True,
            )
            ingest_mode = st.radio(
                "Content type",
                options=["Document-like", "Chat history"],
                index=0,
                horizontal=True,
                help="Documents use minimal deduplication; chat history uses stronger deduplication.",
            )
            chunk_size = st.slider("Chunk size", min_value=300, max_value=3000, value=st.session_state["chunk_size"], step=100)
            overlap = st.slider("Chunk overlap", min_value=0, max_value=800, value=st.session_state["overlap"], step=50)
            build_graph = st.checkbox("Build semantic graph during ingest", value=True)
            st.markdown(
                """
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.session_state["chunk_size"] = chunk_size
        st.session_state["overlap"] = overlap

        manual_text = st.text_area(
            "Paste text here",
            height=240,
            placeholder="Paste notes, article text, or any content you want to store as memory chunks.",
            key="ingest_manual_text",
        )

        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file",
            type=["pdf", "txt"],
            accept_multiple_files=False,
            key="ingest_upload",
        )

        action_col_left, action_col_right = st.columns(2)
        store_clicked = action_col_left.button("Store into Memory")
        clear_preview_clicked = action_col_right.button("Reset preview")

        if clear_preview_clicked:
            st.session_state.pop("last_ingest_summary", None)
            st.session_state.pop("last_graph_state", None)
            st.session_state["ingest_running"] = False
            st.rerun()

        if store_clicked:
            memory_agent = MemoryAgent()
            selected_memory_type = "chat_message" if ingest_mode == "Chat history" else "document_chunk"
            selected_source_type = "chat" if ingest_mode == "Chat history" else "document"

            try:
                source_kind, extracted, source_name = extract_input_text(uploaded_file, manual_text)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            if source_kind == "pdf":
                pdf_page_records = extracted
                if not pdf_page_records:
                    st.warning("No readable text was found in the uploaded PDF.")
                    st.stop()
            elif not extracted.strip():
                st.warning("Please paste text or upload a non-empty TXT file.")
                st.stop()

            chunk_records = prepare_chunk_records(source_kind, extracted, source_name, selected_source_type)
            if not chunk_records:
                st.warning("No chunks were created from the provided content.")
                st.stop()

            with st.spinner("Building embeddings..."):
                memory_blobs = build_memory_blobs(chunk_records, selected_memory_type)

            st.session_state["ingest_running"] = True
            graph_state, stored_count, skipped_count = run_live_ingest(
                memory_agent,
                memory_blobs,
                build_graph=build_graph,
                graph_placeholder=graph_placeholder,
                status_placeholder=status_placeholder,
                progress_placeholder=progress_placeholder,
            )
            st.session_state["ingest_running"] = False

            summary = {
                "stored_count": stored_count,
                "skipped_count": skipped_count,
                "total_chunks": len(memory_blobs),
                "node_count": len(graph_state["entities"]),
                "edge_count": len(graph_state["relationships"]),
                "build_graph": build_graph,
                "stage_text": graph_state.get("last_stage", "Latest stored graph"),
            }
            st.session_state["last_ingest_summary"] = summary
            st.session_state["last_graph_state"] = graph_state

            st.success(f"Stored {stored_count} chunks from {source_name}.")
            st.markdown(
                "<div class='footer-note'>The preview above is driven by the same canonical graph data that was written to Neo4j. Duplicates were skipped before storage.</div>",
                unsafe_allow_html=True,
            )


def render_query_tab():
    st.markdown(
        """
        <div class="query-shell">
            <div class="query-title">Query memory</div>
            <div class="query-copy">Ask a question and the app will retrieve memory, then generate an answer only from the retrieved output.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("query_form"):
        query_depth = st.slider("Query depth", min_value=0, max_value=3, value=0, step=1)
        query_text = st.text_input(
            "Query",
            placeholder="Ask something like: What does hiyansh have to do?",
            key="query_text_input",
        )
        query_submitted = st.form_submit_button("Run Query")

    if query_submitted:
        if not query_text.strip():
            st.warning("Please enter a query first.")
        else:
            memory_agent = MemoryAgent()
            with st.spinner("Retrieving memory..."):
                query_output = memory_agent.retrieve_memory(query_text, depth=query_depth)

            with st.spinner("Generating answer from retrieved memory..."):
                answer_output = answer_query_from_retrieval(query_text, query_output)

            st.markdown(
                """
                <div class="query-results">
                    <div class="query-box">
                        <h4>Retrieved output</h4>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='query-output'><pre>{html.escape(query_output)}</pre></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                    </div>
                    <div class="query-box">
                        <h4>LLM answer</h4>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='query-output'><pre>{html.escape(answer_output)}</pre></div>",
                unsafe_allow_html=True,
            )
            st.markdown("""
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_chat_tab():
    st.markdown(
        """
        <div class="chat-shell">
            <div class="chat-header">
                <div>
                    <div class="query-title">Chat with memory</div>
                    <div class="query-copy">Talk to the assistant while MemLayer retrieves relevant memories and writes the conversation back into memory in the background.</div>
                </div>
                <div class="chat-badge">Live retrieval + background memory ingest</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

        
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("chat_depth", 1)

    action_left, action_right = st.columns([0.72, 0.28])
    with action_left:
        chat_depth = st.slider("Memory depth", min_value=0, max_value=3, value=st.session_state["chat_depth"], step=1)
    with action_right:
        clear_chat = st.button("Clear chat history")

    st.session_state["chat_depth"] = chat_depth

    if clear_chat:
        st.session_state["chat_messages"] = []
        st.rerun()

    chat_messages = st.session_state["chat_messages"]

    if not chat_messages:
        st.markdown(
            """
            <div class="chat-empty">
                Ask a question and I’ll answer using the current memory graph plus any conversation memories that get written while we talk.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in chat_messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                meta = []
                confidence = message.get("retrieval_confidence", {}) or {}
                if confidence:
                    meta.append(
                        f"memory confidence: {confidence.get('label', 'unknown')} ({float(confidence.get('score', 0.0) or 0.0):.2f})"
                    )
                if message.get("memory_ingest_status"):
                    meta.append(message["memory_ingest_status"])
                if meta:
                    st.caption(" | ".join(meta))

    user_input = st.chat_input("Ask about your memories, decisions, projects, or anything you want the assistant to remember.")
    if user_input:
        prior_messages = list(st.session_state["chat_messages"])
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        memory_agent = MemoryAgent()

        with st.spinner("Retrieving memories..."):
            raw_results = memory_agent.retrieve_memory_raw(user_input, depth=chat_depth)
            memory_context = format_chat_memory_context(raw_results)

        with st.spinner("Generating reply..."):
            assistant_output = answer_chat_with_memory(
                user_input,
                memory_context,
                prior_messages,
                raw_results.get("retrieval_confidence", {}),
            )

        retrieval_confidence = raw_results.get("retrieval_confidence", {}) or {}
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": assistant_output,
                "retrieval_confidence": retrieval_confidence,
                "memory_ingest_status": "memory ingest queued in background",
            }
        )

        queue_chat_memory_ingest(user_input, assistant_output, raw_results)
        st.rerun()


def render_reset_tab():
    st.markdown(
        """
        <div class="reset-shell">
            <div class="reset-title">Memory reset</div>
            <div class="reset-copy">These actions clear stored memory from the backing stores. Use them carefully.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="reset-grid">
            <div class="reset-card">
                <h4>Graph memory</h4>
                <p>Deletes all Neo4j nodes and relationships written by the semantic graph.</p>
            </div>
            <div class="reset-card">
                <h4>SQLite memory</h4>
                <p>Clears the episodic memory table used for chunk storage and retrieval.</p>
            </div>
            <div class="reset-card">
                <h4>Qdrant memory</h4>
                <p>Removes every vector point from the episodic collection.</p>
            </div>
            <div class="reset-card">
                <h4>All stores</h4>
                <p>Runs all three clear operations in one pass.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_left, action_mid, action_right, action_all = st.columns(4)

    if action_left.button("Clear Graph Memory"):
        clear_graph_memory()
        st.success("Graph memory cleared.")

    if action_mid.button("Clear DB Memory"):
        clear_sqlite_memory()
        st.success("DB memory cleared.")

    if action_right.button("Clear Qdrant Memory"):
        clear_qdrant_memory()
        st.success("Qdrant memory cleared.")

    if action_all.button("Clear All Memory"):
        clear_all_memory()
        st.success("All memory cleared.")


def main():
    st.set_page_config(
        page_title="MemLayer Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    st.session_state.setdefault("ingest_running", False)
    render_top_hero()

    ingest_tab, query_tab, chat_tab, reset_tab = st.tabs(["Ingest", "Query", "Chat", "Reset"])

    with ingest_tab:
        render_ingest_tab()
    with query_tab:
        render_query_tab()
    with chat_tab:
        render_chat_tab()
    with reset_tab:
        render_reset_tab()


if __name__ == "__main__":
    main()
