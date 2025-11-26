from src.workflows.user_workflow import build_user_analytics_graph

def main():
    # build graph
    graph = build_user_analytics_graph()

    # invoke graph with a test tag (WITHOUT the '#')
    state = graph.invoke({"player_tag": "8C8JJQLG"})

    # inspect outputs
    print("\n--- STATE KEYS ---")
    print(state.keys())

    print("\n--- RAW BATTLES COUNT ---")
    print(len(state.get("battles_raw", [])))

    print("\n--- FILTERED BATTLES COUNT ---")
    print(len(state.get("battles_filtered", [])))

    print("\n--- ANALYTICS SUMMARY KEYS ---")
    if "user_analytics" in state:
        print(state["user_analytics"].keys())

if __name__ == "__main__":
    main()
