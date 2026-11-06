"""
mcp_server/server.py
====================
Model Context Protocol (MCP) server implementation for Cricket Log Semantic Search.
Uses official mcp 2.x MCPServer SDK to expose discoverable tools.
Delegates all operational logic via HTTP to the Django REST Framework backend.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Optional
from mcp.server.mcpserver import MCPServer

# Support both module and direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .client import DjangoAPIClient
except ImportError:
    from mcp_server.client import DjangoAPIClient

logger = logging.getLogger(__name__)

# Initialize MCP Server using official mcp 2.x SDK
mcp_server = MCPServer(
    name="cricket-semantic-search",
    instructions=(
        "You are an expert AI cricket assistant with access to authoritative match event logs. "
        "Use search_cricket_logs for hybrid (BM25 + vector + reranking) search over events and commentary "
        "(including extras, wide, no ball, stumpings, wickets, and boundaries), "
        "query_cricket_stats or get_match_summary for exact numerical runs/event statistics directly calculated from MongoDB, "
        "and ask_cricket_assistant for natural language Q&A with strict evidence grounding and citations."
    ),
)

api_client = DjangoAPIClient()


@mcp_server.tool()
def search_cricket_logs(
    query: str,
    limit: int = 5,
    match_id: str = "",
    team: str = "",
    player: str = "",
    event: str = "",
    status: str = "",
) -> str:
    """
    Search cricket match events using hybrid BM25 + dense vector similarity with reranking.
    
    Args:
        query: Natural language search string (e.g., 'extra runs conceded', 'LBW review in late overs')
        limit: Max candidate records to return (1 to 20)
        match_id: Optional filter by match ID (e.g., 'MATCH001')
        team: Optional filter by team (e.g., 'Team A')
        player: Optional filter by player (e.g., 'Player 9')
        event: Optional filter by event (e.g., 'extras', 'wide', 'no ball', 'stumping', 'wicket taken')
        status: Optional filter by status (e.g., 'verified', 'confirmed', 'disputed')
    """
    res = api_client.semantic_search(
        query=query,
        limit=limit,
        match_id=match_id if match_id else None,
        team=team if team else None,
        player=player if player else None,
        event=event if event else None,
        status=status if status else None,
    )
    return json.dumps(res, indent=2)


@mcp_server.tool()
def query_cricket_stats(
    match_id: str = "",
    team: str = "",
    player: str = "",
    event: str = "",
    status: str = "",
) -> str:
    """
    Calculates exact, authoritative statistics directly from MongoDB.
    
    Args:
        match_id: Match ID (e.g., 'MATCH001')
        team: Team name (e.g., 'Team B')
        player: Player name (e.g., 'Player 9')
        event: Event name (e.g., 'extras', 'stumping', 'wicket taken')
        status: Status string (e.g., 'verified', 'confirmed')
    """
    res = api_client.exact_query(
        match_id=match_id if match_id else None,
        team=team if team else None,
        player=player if player else None,
        event=event if event else None,
        status=status if status else None,
    )
    return json.dumps(res, indent=2)


@mcp_server.tool()
def ask_cricket_assistant(
    question: str,
    limit: int = 5,
) -> str:
    """
    Ask any natural language question about the cricket dataset.
    Routes intelligently to exact database aggregations or Gemini grounded RAG.
    
    Args:
        question: User query (e.g., 'What happened to Player 6 in disputed matches?')
        limit: Number of evidence events to consider (1 to 10)
    """
    res = api_client.chat(query=question, limit=limit)
    return json.dumps(res, indent=2)


@mcp_server.tool()
def get_match_summary(match_id: str) -> str:
    """
    Returns exact total runs, event counts, and team scores for a specified match.
    
    Args:
        match_id: The match identifier (e.g., 'MATCH001')
    """
    res = api_client.exact_query(match_id=match_id)
    return json.dumps(res, indent=2)


def run_mcp_server():
    """Runs the MCP server over stdio transport."""
    mcp_server.run()


if __name__ == "__main__":
    run_mcp_server()
