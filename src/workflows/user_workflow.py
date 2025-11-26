from typing import Any, Dict, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from src.api.battles import get_player_battlelog


# ---------- LangGraph State Definition ----------


class UserAnalyticsState(TypedDict, total=False):
    """
    State for the Phase 1 user analytics workflow.

    Fields (per spec):

      - player_tag: input from user (e.g. "#8C8JJQLG")
      - battles_raw: raw battlelog list from Clash Royale API
      - battles_filtered: ranked/Trophy Road 1v1 battles (normalized)
      - user_analytics: analytics dict (same schema as meta_analytics)
      - user_plots: optional dict of plot paths (usually analytics["plots"])
      - notes: optional list of debug / info strings
    """

    player_tag: str
    battles_raw: List[Dict[str, Any]]
    battles_filtered: List[Dict[str, Any]]
    user_analytics: Dict[str, Any]
    user_plots: Dict[str, Any]
    notes: List[str]


# ---------- Node: fetch_battlelog ----------


def fetch_battlelog_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    LangGraph node:
      - reads player_tag from state
      - calls Clash Royale API
      - writes battles_raw into state
    """
    player_tag = state.get("player_tag")
    if not player_tag:
        raise ValueError("player_tag is required in state for fetch_battlelog_node")

    raw_battles = get_player_battlelog(player_tag)

    # Optional note for debugging / future UI
    note = f"Fetched {len(raw_battles)} battles for {player_tag}"

    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "battles_raw": raw_battles,
        "notes": existing_notes,
    }


# ---------- Graph Builder ----------


def build_user_analytics_graph():
    """
    Build the Phase 1 LangGraph workflow.

    For now (Step D1) this graph only has:
      Start -> fetch_battlelog -> END

    We will extend it in later steps with:
      - filter_and_normalize node
      - compute_user_analytics node
      - generate_user_plots node
    """
    graph = StateGraph(UserAnalyticsState)

    # Nodes
    graph.add_node("fetch_battlelog", fetch_battlelog_node)

    # Entry / edges
    graph.set_entry_point("fetch_battlelog")
    graph.add_edge("fetch_battlelog", END)

    return graph.compile()
