import os
import uuid
import re
from datetime import datetime, timezone

# Project Memory Directories
_project_dir = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(_project_dir, "memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.txt")
MEMORY_LOG_FILE = os.path.join(MEMORY_DIR, "my_memory_log.txt")
PROTOCOL_FILE = os.path.join(_project_dir, "protocol.txt")

os.makedirs(MEMORY_DIR, exist_ok=True)


def run_store_logic(text: str, category: str = "fact") -> str:
    """Stores a new memory entry into memory.txt and my_memory_log.txt."""
    if not text or not text.strip():
        return "Empty text provided. Nothing stored."
    
    mem_id = uuid.uuid4().hex[:12]
    now_iso = datetime.now(timezone.utc).isoformat()
    entry = f"[{mem_id}|{category}|{now_iso}] {text.strip()}\n---\n"
    
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        with open(MEMORY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{now_iso}] STORED ({category}): {text.strip()}\n")
        return f"Successfully stored long-term memory [{mem_id}]: {text.strip()}"
    except Exception as e:
        return f"Error storing memory: {e}"


def run_retrieve_logic(query: str, context: str = "") -> str:
    """Searches memory.txt for entries relevant to query and context."""
    if not os.path.exists(MEMORY_FILE):
        return "No long-term memories found."
    
    query_clean = re.sub(r'[^\w\s]', ' ', (query + " " + context)).lower()
    words = [w for w in query_clean.split() if len(w) > 2]
    
    # Ignore pure greeting words
    greetings = {"hi", "hello", "hey", "greetings", "morning", "evening", "afternoon", "yo", "edith"}
    if words and all(w in greetings for w in words):
        return "Greeting acknowledged. No memory retrieval needed."
    
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        raw_blocks = content.split("\n---\n")
        scored_blocks = []
        
        for block in raw_blocks:
            block_clean = block.strip()
            if not block_clean:
                continue
            block_lower = block_clean.lower()
            score = 0
            for w in words:
                if w in block_lower:
                    score += 1
            if score > 0:
                scored_blocks.append((score, block_clean))
        
        # Sort by relevance score descending
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        top_matches = [b[1] for b in scored_blocks[:5]]
        
        if not top_matches:
            return f"No memories found specifically matching '{query}'."
        
        result_text = "Retrieved Memories:\n" + "\n".join(f"- {m}" for m in top_matches)
        return result_text
    except Exception as e:
        return f"Error retrieving memory: {e}"


def run_update_protocol_logic(update: str) -> str:
    """Updates protocol strategy in memory and protocol.txt."""
    if not update or not update.strip():
        return "Empty update provided."
    
    mem_id = uuid.uuid4().hex[:12]
    now_iso = datetime.now(timezone.utc).isoformat()
    entry = f"[{mem_id}|protocol|{now_iso}] {update.strip()}\n---\n"
    
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        with open(PROTOCOL_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{now_iso}] PROTOCOL UPDATE: {update.strip()}\n")
        return f"Successfully updated behavior protocol [{mem_id}]: {update.strip()}"
    except Exception as e:
        return f"Error updating protocol: {e}"
