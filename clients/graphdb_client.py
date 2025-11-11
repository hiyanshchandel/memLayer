import neo4j
from neo4j import GraphDatabase
from config import NEO4J_URI

uri = NEO4J_URI
graphdb_client = GraphDatabase.driver(uri, auth=None)


