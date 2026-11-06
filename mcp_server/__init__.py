"""
mcp_server/__init__.py
Model Context Protocol (MCP) server package.
Exposes cricket log search and Q&A tools to AI agents via standard MCP protocol.
"""
from .server import mcp_server, run_mcp_server

__all__ = ["mcp_server", "run_mcp_server"]
