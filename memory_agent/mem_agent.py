from memory_blob.definition import MemoryBlob
from episodic.memory_manager import EpisodicMemoryManager
from semantic.sem_mem_man import GraphMemoryManager

class MemoryAgent:
    def __init__(self):
        pass

    def store_memory(self, MemoryBlob):
        episodic_manager = EpisodicMemoryManager()
        episodic_manager.store_memory(MemoryBlob)
        graph_manager = GraphMemoryManager()
        graph_manager.push_to_graphdb(MemoryBlob)
    
    def retrieve_memory(self, query):
        episodic_manager = EpisodicMemoryManager()
        return episodic_manager.retrieve_memory(query)
        return