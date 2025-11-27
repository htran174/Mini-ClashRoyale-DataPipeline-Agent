# ============================================================
# test_meta_graph.py  (DROP-IN)
# ============================================================

from src.workflows.meta_workflow import build_meta_graph


initial_state = {
    "top_players": [],
    "required_deck_types": [],
    "selected_players": [],
    "used_player_indices": set(),
    "fetched_player_tags": set(),
    "meta_raw_battles": [],
    "normalized_battles": [],
    "deck_type_counts": {},
    "is_balanced": False,
    "loop_count": 0,
    "notes": [],
}


if __name__ == "__main__":
    graph = build_meta_graph()
    state = graph.invoke(initial_state)

    print("Top players:", len(state.get("top_players", [])))
    print("Selected players:", len(state.get("selected_players", [])))
    print("Fetched tags:", len(state.get("fetched_player_tags", [])))
    print("Meta raw battles:", len(state.get("meta_raw_battles", [])))
    print(state["deck_type_counts"])

    print("\nNotes:")
    for note in state.get("notes", []):
        print(" -", note)
