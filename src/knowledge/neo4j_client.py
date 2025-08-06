"""
Neo4j Client for Legal Contract Knowledge Graph
Simple wrapper for Neo4j operations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

try:
    from neo4j import GraphDatabase, Transaction

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j package not installed. Run: pip install neo4j")

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Simple Neo4j client for legal contract knowledge graph operations."""

    def __init__(self, config_path: str = "config/graph_config.json"):
        """Initialize Neo4j client with configuration."""

        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package is required. Install with: pip install neo4j"
            )

        self.config = self._load_config(config_path)
        self.driver = None
        self._connect()

    def _load_config(self, config_path: str) -> Dict:
        """Load Neo4j configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            return json.load(f)

    def _connect(self):
        """Connect to Neo4j database."""
        try:
            neo4j_config = self.config["neo4j"]
            self.driver = GraphDatabase.driver(
                neo4j_config["uri"],
                auth=(neo4j_config["username"], neo4j_config["password"]),
            )

            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")

            logger.info(f"Connected to Neo4j at {neo4j_config['uri']}")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        try:
            with self.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise

    def create_indexes(self):
        """Create indexes based on schema configuration."""
        schema = self.config["graph_schema"]

        for node_type, node_config in schema["nodes"].items():
            indexes = node_config.get("indexes", [])

            for index_property in indexes:
                query = f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.{index_property})"
                try:
                    self.execute_query(query)
                    logger.info(f"Created index: {node_type}.{index_property}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create index {node_type}.{index_property}: {e}"
                    )

    def clear_graph(self, confirm: bool = False):
        """Clear all nodes and relationships from the graph."""
        if not confirm:
            logger.warning("Clear operation requires confirm=True")
            return

        logger.info("Clearing entire graph...")
        self.execute_query("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    def create_contract_node(self, contract_data: Dict) -> str:
        """Create a contract node."""
        query = """
        MERGE (c:Contract {contract_id: $contract_id})
        SET c.title = $title,
            c.created_at = $created_at,
            c.updated_at = datetime()
        RETURN c.contract_id as contract_id
        """

        result = self.execute_query(query, contract_data)
        return result[0]["contract_id"] if result else None

    def create_party_node(self, party_data: Dict) -> str:
        """Create a party node."""
        query = """
        MERGE (p:Party {party_id: $party_id})
        SET p.name = $name,
            p.entity_type = $entity_type,
            p.updated_at = datetime()
        RETURN p.party_id as party_id
        """

        result = self.execute_query(query, party_data)
        return result[0]["party_id"] if result else None

    def create_clause_node(self, clause_data: Dict) -> str:
        """Create a clause node."""
        query = """
        MERGE (cl:Clause {clause_id: $clause_id})
        SET cl.clause_type = $clause_type,
            cl.original_text = $original_text,
            cl.coverage_score = $coverage_score,
            cl.updated_at = datetime()
        RETURN cl.clause_id as clause_id
        """

        result = self.execute_query(query, clause_data)
        return result[0]["clause_id"] if result else None

    def create_template_node(self, template_data: Dict) -> str:
        """Create a template node."""
        query = """
        MERGE (t:Template {template_id: $template_id})
        SET t.template_name = $template_name,
            t.priority = $priority,
            t.generated_at = $generated_at,
            t.coverage_score = $coverage_score,
            t.clause_type = $clause_type,
            t.original_text_preserved = $original_text_preserved,
            t.file_path = $file_path,
            t.updated_at = datetime()
        RETURN t.template_id as template_id
        """

        result = self.execute_query(query, template_data)
        return result[0]["template_id"] if result else None

    def create_variable_node(self, variable_data: Dict) -> str:
        """Create a variable node."""
        query = """
        CREATE (v:Variable {
            var_name: $var_name,
            var_type: $var_type,
            value: $value,
            extracted_value: $extracted_value,
            created_at: datetime()
        })
        RETURN id(v) as variable_id
        """

        result = self.execute_query(query, variable_data)
        return result[0]["variable_id"] if result else None

    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        from_label: str,
        to_label: str,
        properties: Dict = None,
    ):
        """Create a relationship between two nodes."""

        # Build the query dynamically based on labels
        query = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE a.{self._get_id_field(from_label)} = $from_id 
        AND b.{self._get_id_field(to_label)} = $to_id
        MERGE (a)-[r:{rel_type}]->(b)
        """

        if properties:
            set_clauses = [f"r.{key} = ${key}" for key in properties.keys()]
            query += f" SET {', '.join(set_clauses)}"

        parameters = {"from_id": from_id, "to_id": to_id, **(properties or {})}

        self.execute_query(query, parameters)

    def _get_id_field(self, label: str) -> str:
        """Get the ID field name for a node label."""
        id_fields = {
            "Contract": "contract_id",
            "Party": "party_id",
            "Clause": "clause_id",
            "Variable": "var_name",  # Variables use name as ID
            "Template": "template_id",
        }
        return id_fields.get(label, "id")

    def create_similarity_relationships(self, similarity_data: List[Dict]):
        """Create similarity relationships between clauses in batch."""
        query = """
        UNWIND $similarity_data as sim
        MATCH (c1:Clause {clause_id: sim.clause1_id})
        MATCH (c2:Clause {clause_id: sim.clause2_id})
        MERGE (c1)-[r:SIMILAR_TO]->(c2)
        SET r.similarity_score = sim.score
        """

        self.execute_query(query, {"similarity_data": similarity_data})
        logger.info(f"Created {len(similarity_data)} similarity relationships")

    def get_stats(self) -> Dict:
        """Get basic graph statistics."""
        queries = {
            "contracts": "MATCH (c:Contract) RETURN count(c) as count",
            "parties": "MATCH (p:Party) RETURN count(p) as count",
            "clauses": "MATCH (cl:Clause) RETURN count(cl) as count",
            "variables": "MATCH (v:Variable) RETURN count(v) as count",
            "templates": "MATCH (t:Template) RETURN count(t) as count",
            "relationships": "MATCH ()-[r]->() RETURN count(r) as count",
        }

        stats = {}
        for stat_name, query in queries.items():
            try:
                result = self.execute_query(query)
                stats[stat_name] = result[0]["count"] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get {stat_name} count: {e}")
                stats[stat_name] = 0

        return stats

    def run_sample_queries(self) -> Dict:
        """Run some sample queries to test the graph."""
        sample_queries = {
            "top_parties": """
                MATCH (p:Party)-[:INVOLVES]-(c:Contract)
                RETURN p.name, count(c) as contract_count
                ORDER BY contract_count DESC
                LIMIT 5
            """,
            "clause_types": """
                MATCH (cl:Clause)
                RETURN cl.clause_type, count(cl) as count
                ORDER BY count DESC
            """,
            "template_success_rate": """
                MATCH (cl:Clause)
                OPTIONAL MATCH (cl)-[:GENERATES]->(t:Template)
                WITH cl.clause_type as clause_type, 
                     count(cl) as total_clauses,
                     count(t) as templates_generated
                WHERE total_clauses > 0
                RETURN clause_type, 
                       total_clauses,
                       templates_generated,
                       round(100.0 * templates_generated / total_clauses) as success_rate
                ORDER BY success_rate DESC
                LIMIT 10
            """,
            "similar_clauses": """
                MATCH (c1:Clause)-[s:SIMILAR_TO]->(c2:Clause)
                RETURN c1.clause_type, c2.clause_type, s.similarity_score
                ORDER BY s.similarity_score DESC
                LIMIT 5
            """,
        }

        results = {}
        for query_name, query in sample_queries.items():
            try:
                results[query_name] = self.execute_query(query)
            except Exception as e:
                logger.warning(f"Sample query {query_name} failed: {e}")
                results[query_name] = []

        return results
