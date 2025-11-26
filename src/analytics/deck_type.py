import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- Load card metadata ----------

BASE_DIR = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = BASE_DIR / "data"
CARD_METADATA_PATH = DATA_DIR / "card_metadata.json"

with CARD_METADATA_PATH.open("r", encoding="utf-8") as f:
    _CARD_META_LIST: List[Dict[str, Any]] = json.load(f)

# Map by card name for quick lookup
_CARD_META_BY_NAME: Dict[str, Dict[str, Any]] = {c["name"]: c for c in _CARD_META_LIST}


def _get_card_meta(card_name: str) -> Dict[str, Any]:
    """Safely get metadata for a card by name."""
    return _CARD_META_BY_NAME.get(card_name, {})


# ---------- Archetype constants ----------

ARCHETYPE_SIEGE = "Siege"
ARCHETYPE_BAIT = "Bait"
ARCHETYPE_CYCLE = "Cycle"
ARCHETYPE_BRIDGE_SPAM = "Bridge Spam"
ARCHETYPE_BEATDOWN = "Beatdown"
ARCHETYPE_HYBRID = "Hybrid"

# For Siege rules
_SIEGE_XBOW = {"X-Bow"}
_SIEGE_MORTAR = {"Mortar"}


def _precompute_deck_values(cards: List[str]) -> Dict[str, Any]:
    """
    Compute:
      - avg_elixir
      - four_card_cycle_cost
      - has_xbow / has_mortar
      - bait_pieces (using is_bait_piece or explicit names)
      - has_bait_core (Goblin Barrel + at least 1 other bait piece)
      - bridge_spam_count (using is_bridge_spam_piece)
      - big_tank_count (using is_big_tank)
    """
    metas = [_get_card_meta(c) for c in cards]

    elixirs: List[float] = [
        m["elixir"] for m in metas if isinstance(m.get("elixir"), (int, float))
    ]
    if len(elixirs) == 0:
        avg_elixir = 3.0
        four_cycle = 12.0
    else:
        # avg elixir
        avg_elixir = sum(elixirs) / 8.0  # deck = 8 cards (some safety if metadata missing)
        # four-card cycle cost
        four_cycle = sum(sorted(elixirs)[:4])

    names_set = set(cards)

    has_xbow = len(names_set & _SIEGE_XBOW) > 0
    has_mortar = len(names_set & _SIEGE_MORTAR) > 0

    # bait_pieces – primarily from metadata flag
    bait_pieces = sum(1 for m in metas if m.get("is_bait_piece"))

    # Goblin Barrel is a bait card even if metadata flag is False
    has_goblin_barrel = "Goblin Barrel" in names_set
    if has_goblin_barrel:
        # B1 rule needs: barrel + at least one other bait piece
        has_bait_core = bait_pieces >= 1
    else:
        has_bait_core = False

    bridge_spam_count = sum(1 for m in metas if m.get("is_bridge_spam_piece"))
    big_tank_count = sum(1 for m in metas if m.get("is_big_tank"))

    return {
        "avg_elixir": avg_elixir,
        "four_card_cycle_cost": four_cycle,
        "has_xbow": has_xbow,
        "has_mortar": has_mortar,
        "bait_pieces": bait_pieces,
        "has_bait_core": has_bait_core,
        "bridge_spam_count": bridge_spam_count,
        "big_tank_count": big_tank_count,
    }


def classify_deck(cards: List[str]) -> str:
    """
    Classify a deck into one of your archetypes.

    Priority order (first match wins):
      1) Siege
      2) Bait
      3) Cycle
      4) Bridge Spam
      5) Beatdown
      6) Hybrid (fallback)
    """
    if not cards:
        return ARCHETYPE_HYBRID

    v = _precompute_deck_values(cards)

    avg_elixir = v["avg_elixir"]
    four_cycle = v["four_card_cycle_cost"]
    has_xbow = v["has_xbow"]
    has_mortar = v["has_mortar"]
    bait_pieces = v["bait_pieces"]
    has_bait_core = v["has_bait_core"]
    bridge_spam_count = v["bridge_spam_count"]
    big_tank_count = v["big_tank_count"]

    # =========================================
    # 1️⃣ SIEGE RULES
    # =========================================
    # S1: X-Bow hard rule
    if has_xbow:
        return ARCHETYPE_SIEGE

    # S2: Mortar hard rule
    if has_mortar:
        return ARCHETYPE_SIEGE

    # =========================================
    # 2️⃣ BAIT RULES (PACKAGE-BASED)
    # =========================================
    # B1: Goblin Barrel + at least one other bait unit
    if has_bait_core and bait_pieces >= 1:
        return ARCHETYPE_BAIT

    # =========================================
    # 3️⃣ CYCLE RULES (4-card cycle cost)
    # =========================================
    # CY1: If four_card_cycle_cost <= 9 -> Cycle
    if four_cycle <= 9:
        return ARCHETYPE_CYCLE

    # =========================================
    # 4️⃣ BRIDGE SPAM RULES (key piece count)
    # =========================================
    # BS1: If bridge_spam_count >= 2 -> Bridge Spam
    if bridge_spam_count >= 2:
        return ARCHETYPE_BRIDGE_SPAM

    # =========================================
    # 5️⃣ BEATDOWN RULES (tank + heavy avg)
    # =========================================
    # BD1: If big_tank_count >= 1 AND avg_elixir >= 3.5 -> Beatdown
    if big_tank_count >= 1 and avg_elixir >= 3.5:
        return ARCHETYPE_BEATDOWN

    # =========================================
    # 6️⃣ HYBRID (fallback)
    # =========================================
    return ARCHETYPE_HYBRID


# ---------- Aggregation over battles ----------


def _init_type_bucket(deck_type: str) -> Dict[str, Any]:
    return {
        "type": deck_type,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "win_rate": 0.0,
    }


def _finalize_stats(raw: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert {deck_type: stats} into a sorted list, computing win_rate (0–1).
    Sort by games desc.
    """
    result: List[Dict[str, Any]] = []
    for deck_type, s in raw.items():
        games = s["games"]
        wins = s["wins"]
        losses = s["losses"]
        draws = s["draws"]
        win_rate = wins / games if games > 0 else 0.0
        result.append(
            {
                "type": deck_type,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
            }
        )

    result.sort(key=lambda d: d["games"], reverse=True)
    return result


def summarize_deck_types(
    battles: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Given normalized battles, compute deck-type stats for:

      - my_deck_types
      - opp_deck_types

    Each battle is expected to have:
        - "my_cards": List[str]
        - "opp_cards": List[str]
        - "result": "win" | "loss" | "draw"

    Returns:
        (my_deck_types_list, opp_deck_types_list)
    """
    my_raw: Dict[str, Dict[str, Any]] = {}
    opp_raw: Dict[str, Dict[str, Any]] = {}

    for b in battles:
        my_cards = b.get("my_cards") or []
        opp_cards = b.get("opp_cards") or []
        result = (b.get("result") or "").lower()

        my_type = classify_deck(my_cards)
        opp_type = classify_deck(opp_cards)

        if my_type not in my_raw:
            my_raw[my_type] = _init_type_bucket(my_type)
        if opp_type not in opp_raw:
            opp_raw[opp_type] = _init_type_bucket(opp_type)

        my_raw[my_type]["games"] += 1
        opp_raw[opp_type]["games"] += 1

        if result == "win":
            # I won this game
            my_raw[my_type]["wins"] += 1
            opp_raw[opp_type]["wins"] += 1
        elif result == "loss":
            # I lost this game
            my_raw[my_type]["losses"] += 1
            opp_raw[opp_type]["losses"] += 1
        else:
            # draw
            my_raw[my_type]["draws"] += 1
            opp_raw[opp_type]["draws"] += 1

    my_types = _finalize_stats(my_raw)
    opp_types = _finalize_stats(opp_raw)

    return my_types, opp_types
