# ============================================================
# meta_workflow.py  (DROP-IN REPLACEMENT)
# ============================================================

from typing import Any, Dict, List, Set
from typing_extensions import TypedDict

import random

from langgraph.graph import StateGraph, END

from src.api.players import fetch_top_300_players
from src.api.battles import get_player_battlelog
from src.analytics.battle_filters import filter_and_normalize_ranked_1v1


# ============================================================
# Phase 0 State Definition
# ============================================================

class MetaState(TypedDict, total=False):
    top_players: List[Dict[str, Any]]

    required_deck_types: List[str]

    selected_players: List[Dict[str, Any]]
    used_player_indices: Set[int]
    fetched_player_tags: Set[str]

    meta_raw_battles: List[Dict[str, Any]]
    normalized_battles: List[Dict[str, Any]]  # (filled in later)
    deck_type_counts: Dict[str, int]  # (filled in later)

    is_balanced: bool
    loop_count: int

    notes: List[str]


# ============================================================
# Node: fetch_top_300
# ============================================================

def fetch_top_300_node(state: MetaState) -> Dict[str, Any]:
    players = fetch_top_300_players()

    return {
        "top_players": players,
        "required_deck_types": [
            "Siege",
            "Bait",
            "Cycle",
            "Bridge Spam",
            "Beatdown"
            # Hybrid excluded on purpose
        ],
        "selected_players": [],
        "used_player_indices": set(),
        "fetched_player_tags": set(),
        "meta_raw_battles": [],
        "normalized_battles": [],
        "deck_type_counts": {},
        "is_balanced": False,
        "loop_count": 0,
        "notes": [f"Fetched {len(players)} top players from API"],
    }


# ============================================================
# Node: sample_initial_50
# ============================================================

def sample_initial_50_node(state: MetaState) -> Dict[str, Any]:
    top_players = state["top_players"]
    used = set(state["used_player_indices"])

    available_indices = [i for i in range(len(top_players)) if i not in used]

    if len(available_indices) < 50:
        sample_indices = available_indices
    else:
        sample_indices = random.sample(available_indices, 50)

    sampled_players = [top_players[i] for i in sample_indices]
    new_used = used.union(sample_indices)

    note = f"sample_initial_50: sampled {len(sample_indices)} players out of {len(top_players)}."

    return {
        "selected_players": sampled_players,
        "used_player_indices": new_used,
        "notes": state["notes"] + [note],
    }


# ============================================================
# Node: fetch_meta_battles  (using Phase 1 function)
# ============================================================

def fetch_meta_battles_node(state: MetaState) -> Dict[str, Any]:
    selected = state["selected_players"]
    fetched = set(state["fetched_player_tags"])
    meta_raw = list(state["meta_raw_battles"])
    notes = list(state["notes"])

    new_players = 0
    new_battles = 0

    for p in selected:
        tag = p.get("tag")
        if not tag or tag in fetched:
            continue

        new_players += 1

        try:
            raw_log = get_player_battlelog(tag)
        except Exception as e:
            notes.append(f"fetch_meta_battles: error fetching {tag}: {e}")
            continue

        normalized = filter_and_normalize_ranked_1v1(raw_log)
        battles = normalized[:10]

        # annotate for debugging
        for b in battles:
            b["_meta_source_tag"] = tag

        meta_raw.extend(battles)
        new_battles += len(battles)
        fetched.add(tag)

    notes.append(
        f"fetch_meta_battles: fetched {new_battles} normalized ranked 1v1 battles "
        f"from {new_players} new players. total_meta_battles={len(meta_raw)}"
    )

    return {
        "meta_raw_battles": meta_raw,
        "fetched_player_tags": fetched,
        "notes": notes,
    }


# ============================================================
# Graph Construction
# ============================================================

def build_meta_graph():
    graph = StateGraph(MetaState)

    graph.add_node("fetch_top_300", fetch_top_300_node)
    graph.add_node("sample_initial_50", sample_initial_50_node)
    graph.add_node("fetch_meta_battles", fetch_meta_battles_node)

    graph.set_entry_point("fetch_top_300")
    graph.add_edge("fetch_top_300", "sample_initial_50")
    graph.add_edge("sample_initial_50", "fetch_meta_battles")
    graph.add_edge("fetch_meta_battles", END)

    return graph.compile()
