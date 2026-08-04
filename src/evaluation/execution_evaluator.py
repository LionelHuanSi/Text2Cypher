import json
import logging
from typing import Dict, List, Tuple
from configs.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_execution_exact_match(test_executable_samples: List[Dict], predictions: List[str]) -> Tuple[float, List[bool]]:
    """
    Executes student/teacher Cypher vs Ground-truth Cypher on Neo4j databases.
    Only runs on valid test_executable samples (2,471 subset).
    Returns: (Exact Match Accuracy %, Match results per sample)
    """
    logger.info(f"Running Execution-based Exact Match evaluation on {len(test_executable_samples)} executable test samples...")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        logger.warning(f"Could not connect to Neo4j driver at '{NEO4J_URI}': {e}")
        logger.warning("Simulation Mode: Exact match execution will return 0.0 until Neo4j DB is connected.")
        return 0.0, [False] * len(predictions)

    matches = []
    
    for item, pred_cypher in zip(test_executable_samples, predictions):
        db_name = item.get("database_reference_alias") or item.get("database_reference") or "neo4j"
        gt_cypher = item.get("cypher", "")
        
        if not pred_cypher or not gt_cypher:
            matches.append(False)
            continue

        try:
            with driver.session(database=db_name) as session:
                res_pred = session.run(pred_cypher).data()
                res_gt = session.run(gt_cypher).data()

                # Lexicographical sort comparison (Paper Section 4.2)
                sorted_pred = sorted(json.dumps(r, sort_keys=True) for r in res_pred)
                sorted_gt = sorted(json.dumps(r, sort_keys=True) for r in res_gt)

                is_exact = (sorted_pred == sorted_gt)
                matches.append(is_exact)
        except Exception as err:
            matches.append(False)

    driver.close()

    acc = (sum(matches) / len(matches) * 100.0) if matches else 0.0
    logger.info(f"Execution Exact Match Accuracy: {acc:.2f}% ({sum(matches)}/{len(matches)} passed)")
    return round(acc, 2), matches
