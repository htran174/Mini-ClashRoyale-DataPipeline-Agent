from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd

from .deck_type import summarize_deck_types


def build_battles_dataframe(
    battles_normalized: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Turn a list of normalized battle dicts into a pandas DataFrame.

    Expected battle schema:
        {
          "battle_time": str,
          "result": "win" | "loss" | "draw",
          "my_cards": List[str],
          "opp_cards": List[str],
          "mode_name": str,
        }
    """
    if not battles_normalized:
        return pd.DataFrame(
            columns=["battle_time", "result", "my_cards", "opp_cards", "mode_name"]
        )

    df = pd.DataFrame(battles_normalized)

    for col in ["battle_time", "result", "my_cards", "opp_cards", "mode_name"]:
        if col not in df.columns:
            df[col] = None

    return df


# ---------- Overall summary ----------


def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall performance summary."""
    total_games = len(df)
    if total_games == 0:
        return {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
        }

    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    draws = int((df["result"] == "draw").sum())

    win_rate = wins / total_games if total_games > 0 else 0.0

    return {
        "games_played": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
    }


# ---------- Card-level stats ----------


def _card_stats_from_rows(
    rows: List[Dict[str, Any]],
    min_games: int = 3,
    sort_desc: bool = True,
) -> List[Dict[str, Any]]:
    """
    Helper to build card stats from rows like:
        {"card": "Card Name", "result": "win" | "loss" | "draw"}
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )

    for row in rows:
        card = row["card"]
        result = row["result"]
        s = stats[card]
        s["games"] += 1
        if result == "win":
            s["wins"] += 1
        elif result == "loss":
            s["losses"] += 1
        else:
            s["draws"] += 1

    out: List[Dict[str, Any]] = []
    for card, s in stats.items():
        if s["games"] < min_games:
            continue
        wr = s["wins"] / s["games"] if s["games"] > 0 else 0.0
        out.append(
            {
                "card": card,
                "games": s["games"],
                "wins": s["wins"],
                "losses": s["losses"],
                "draws": s["draws"],
                "win_rate": wr,
            }
        )

    out.sort(key=lambda x: (x["win_rate"], x["games"]), reverse=sort_desc)
    return out


def compute_card_performance(df: pd.DataFrame, min_games: int = 3) -> Dict[str, Any]:
    """
    Compute card-level performance for:
      - my cards  (best/worst)
      - opponent cards (tough/easy)
    """
    rows_my: List[Dict[str, Any]] = []
    rows_opp: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        result = row["result"]

        for card in row.get("my_cards", []) or []:
            rows_my.append({"card": card, "result": result})

        for card in row.get("opp_cards", []) or []:
            if result == "win":
                opp_result = "loss"
            elif result == "loss":
                opp_result = "win"
            else:
                opp_result = "draw"
            rows_opp.append({"card": card, "result": opp_result})

    my_stats_desc = _card_stats_from_rows(rows_my, min_games=min_games, sort_desc=True)
    my_stats_asc = list(reversed(my_stats_desc))

    opp_stats_desc = _card_stats_from_rows(
        rows_opp, min_games=min_games, sort_desc=True
    )
    opp_stats_asc = list(reversed(opp_stats_desc))

    return {
        "best_cards": my_stats_desc,
        "worst_cards": my_stats_asc,
        "tough_opp_cards": opp_stats_desc,
        "easy_opp_cards": opp_stats_asc,
    }


# ---------- Deck-level stats (exact deck lists) ----------


def compute_deck_performance(
    battles_normalized: List[Dict[str, Any]], min_games: int = 3
) -> Dict[str, Any]:
    """
    Compute performance by my deck and opponent deck.

    Deck key = sorted tuple of 8 card names.
    """
    my_decks: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    opp_decks: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )

    for b in battles_normalized:
        result = b["result"]
        my_key = tuple(sorted(b.get("my_cards", [])))
        opp_key = tuple(sorted(b.get("opp_cards", [])))

        ms = my_decks[my_key]
        ms["games"] += 1
        if result == "win":
            ms["wins"] += 1
        elif result == "loss":
            ms["losses"] += 1
        else:
            ms["draws"] += 1

        os = opp_decks[opp_key]
        os["games"] += 1
        if result == "win":
            os["losses"] += 1
        elif result == "loss":
            os["wins"] += 1
        else:
            os["draws"] += 1

    def _deck_dicts(
        decks_stats: Dict[Tuple[str, ...], Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for deck_key, s in decks_stats.items():
            if s["games"] < min_games:
                continue
            wr = s["wins"] / s["games"] if s["games"] > 0 else 0.0
            out.append(
                {
                    "deck": list(deck_key),
                    "games": s["games"],
                    "wins": s["wins"],
                    "losses": s["losses"],
                    "draws": s["draws"],
                    "win_rate": wr,
                }
            )
        out.sort(key=lambda x: (x["win_rate"], x["games"]), reverse=True)
        return out

    my_decks_list = _deck_dicts(my_decks)
    opp_decks_list = _deck_dicts(opp_decks)

    return {
        "best_decks": my_decks_list,
        "worst_decks": list(reversed(my_decks_list)),
        "tough_matchups": opp_decks_list,
        "easy_matchups": list(reversed(opp_decks_list)),
    }


# ---------- Main entrypoint ----------


def compute_user_analytics(
    battles_normalized: List[Dict[str, Any]],
    min_card_games: int = 3,
    min_deck_games: int = 3,
) -> Dict[str, Any]:
    """
    Main entrypoint for user analytics (and also meta analytics).

    Returns dict:

        {
          "summary": {...},
          "best_cards": [...],
          "worst_cards": [...],
          "tough_opp_cards": [...],
          "easy_opp_cards": [...],
          "best_decks": [...],
          "worst_decks": [...],
          "tough_matchups": [...],
          "easy_matchups": [...],
          "my_deck_types": [...],
          "opp_deck_types": [...],
          "plots": {...}
        }
    """
    df = build_battles_dataframe(battles_normalized)

    summary = compute_summary(df)
    card_stats = compute_card_performance(df, min_games=min_card_games)
    deck_stats = compute_deck_performance(
        battles_normalized, min_games=min_deck_games
    )

    my_deck_types, opp_deck_types = summarize_deck_types(battles_normalized)

    analytics: Dict[str, Any] = {
        "summary": summary,
        "best_cards": card_stats["best_cards"],
        "worst_cards": card_stats["worst_cards"],
        "tough_opp_cards": card_stats["tough_opp_cards"],
        "easy_opp_cards": card_stats["easy_opp_cards"],
        "best_decks": deck_stats["best_decks"],
        "worst_decks": deck_stats["worst_decks"],
        "tough_matchups": deck_stats["tough_matchups"],
        "easy_matchups": deck_stats["easy_matchups"],
        "my_deck_types": my_deck_types,
        "opp_deck_types": opp_deck_types,
        "plots": {},
    }

    return analytics
