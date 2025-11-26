from typing import Any, Dict, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from src.api.battles import get_player_battlelog
from src.analytics.battle_filters import filter_and_normalize_ranked_1v1
from src.analytics.user_analytics import compute_user_analytics
from src.analytics.plots import generate_card_plots


# ---------- LangGraph State Definition ----------


class UserAnalyticsState(TypedDict, total=False):
    """
    State for the Phase 1 user analytics workflow.

    Fields:

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
    Node:
      - reads player_tag from state
      - calls Clash Royale API
      - writes battles_raw into state
    """
    player_tag = state.get("player_tag")
    if not player_tag:
        raise ValueError("player_tag is required in state for fetch_battlelog_node")

    raw_battles = get_player_battlelog(player_tag)

    note = f"Fetched {len(raw_battles)} battles for {player_tag}"
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "battles_raw": raw_battles,
        "notes": existing_notes,
    }


# ---------- Node: filter_and_normalize ----------


def filter_and_normalize_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads battles_raw
      - filters to ranked/Trophy Road 1v1
      - normalizes battles
      - writes battles_filtered into state
    """
    battles_raw = state.get("battles_raw", []) or []
    if not battles_raw:
        raise ValueError("battles_raw is required in state for filter_and_normalize_node")

    battles_filtered = filter_and_normalize_ranked_1v1(battles_raw)

    note = f"Filtered to {len(battles_filtered)} ranked/Trophy Road 1v1 battles"
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "battles_filtered": battles_filtered,
        "notes": existing_notes,
    }


# ---------- Node: compute_user_analytics ----------


def compute_user_analytics_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads battles_filtered
      - computes analytics dict
      - writes user_analytics into state
    """
    battles_filtered = state.get("battles_filtered", []) or []
    if not battles_filtered:
        raise ValueError(
            "battles_filtered is required in state for compute_user_analytics_node"
        )

    analytics = compute_user_analytics(battles_filtered)

    note = (
        f"Computed analytics on {analytics.get('summary', {}).get('games_played', 0)} "
        "ranked/Trophy Road 1v1 battles"
    )
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "user_analytics": analytics,
        "notes": existing_notes,
    }


# ---------- Node: generate_user_plots ----------


def generate_user_plots_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads user_analytics
      - generates card-level plots
      - writes user_plots and updated user_analytics into state
    """
    analytics = state.get("user_analytics") or {}
    if not analytics:
        raise ValueError(
            "user_analytics is required in state for generate_user_plots_node"
        )

    analytics_with_plots = generate_card_plots(analytics, prefix="user")

    note = "Generated user card-level plots"
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "user_analytics": analytics_with_plots,
        "user_plots": analytics_with_plots.get("plots", {}),
        "notes": existing_notes,
    }


# ---------- Graph Builder ----------


def build_user_analytics_graph():
    """
    Build the Phase 1 LangGraph workflow.

    Pipeline (D4):

        fetch_battlelog
            -> filter_and_normalize
            -> compute_user_analytics
            -> generate_user_plots
            -> END
    """
    graph = StateGraph(UserAnalyticsState)

    # Nodes
    graph.add_node("fetch_battlelog", fetch_battlelog_node)
    graph.add_node("filter_and_normalize", filter_and_normalize_node)
    graph.add_node("compute_user_analytics", compute_user_analytics_node)
    graph.add_node("generate_user_plots", generate_user_plots_node)

    # Entry / edges
    graph.set_entry_point("fetch_battlelog")
    graph.add_edge("fetch_battlelog", "filter_and_normalize")
    graph.add_edge("filter_and_normalize", "compute_user_analytics")
    graph.add_edge("compute_user_analytics", "generate_user_plots")
    graph.add_edge("generate_user_plots", END)

    return graph.compile()
