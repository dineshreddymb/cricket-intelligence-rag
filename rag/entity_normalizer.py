"""
rag/entity_normalizer.py
========================
Generalized entity extraction and natural language query normalization engine.
Handles diverse real-world user inputs:
- Flexible match identifiers (e.g., 'm1', 'match 1', 'game 1', '1st match', 'MATCH001')
- Flexible player identifiers (e.g., 'player 2', 'p2', 'player-2', 'batsman 2')
- Flexible team identifiers (e.g., 'team a', 'team-a', 'teama', 'side a')
- Flexible over identifiers (e.g., 'over 15', '15th over', 'ov 15')
- Cricket event domain ontology mapping with plurals, slang, and hyphenations.
"""

import re
from typing import Optional, Tuple, Dict, Any, List

# Ordinal word mapping for matches and overs
ORDINAL_MAP = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}

# Regex patterns for Matches
RE_MATCH_EXACT = re.compile(r"\b(MATCH\d{3})\b", re.IGNORECASE)
RE_MATCH_NUMBERED = re.compile(
    r"\b(?:MATCH|GAME|M)\s*[-_#]?\s*(?:NO\.?|NUMBER)?\s*(\d{1,3})\b", re.IGNORECASE
)
RE_MATCH_ORDINAL_WORD = re.compile(
    r"\b(" + "|".join(ORDINAL_MAP.keys()) + r")\s+(?:MATCH|GAME)\b", re.IGNORECASE
)

# Regex patterns for Teams
RE_TEAM = re.compile(
    r"\b(?:TEAM|SIDE)\s*[-_]?\s*([A-D])\b|\bTEAM([A-D])\b", re.IGNORECASE
)

# Regex patterns for Players
RE_PLAYER = re.compile(
    r"\b(?:PLAYER|BATSMAN|BOWLER|FIELDER)\s*[-_#]?\s*(\d{1,2})\b|\bP\s*[-_#]?\s*(\d{1,2})\b",
    re.IGNORECASE,
)

# Regex patterns for Overs
RE_OVER_NUMBERED = re.compile(
    r"\b(?:OVER|OV\.?)\s*[-_#]?\s*(\d{1,2})\b", re.IGNORECASE
)
RE_OVER_ORDINAL = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+OVER\b", re.IGNORECASE
)

# Comprehensive Cricket Event Ontology
CANONICAL_EVENT_KEYWORDS: Dict[str, str] = {
    # Extras (wide & no ball)
    "extra runs": "extras",
    "extra run": "extras",
    "extras": "extras",
    "extra": "extras",
    "penalty runs": "extras",
    "illegal delivery": "extras",
    "illegal deliveries": "extras",
    "illegal balls": "extras",
    "illegal ball": "extras",
    "wide or no-ball": "extras",
    "wide or no ball": "extras",
    "wide and no-ball": "extras",
    "wide and no ball": "extras",
    "wides or no-balls": "extras",
    "no-ball": "no ball",
    "no-balls": "no ball",
    "no ball": "no ball",
    "no balls": "no ball",
    "noball": "no ball",
    "noballs": "no ball",
    "wide": "wide",
    "wides": "wide",
    "wide ball": "wide",
    "wide balls": "wide",

    # Dismissals & Wickets
    "stumping or catch-related": "fielding dismissals",
    "stumping or catch": "fielding dismissals",
    "stumpings or catches": "fielding dismissals",
    "catch or stumping": "fielding dismissals",
    "catches or stumpings": "fielding dismissals",
    "stumping": "stumping",
    "stumpings": "stumping",
    "stumped": "stumping",
    "catch taken": "catch taken",
    "catches taken": "catch taken",
    "catch": "catch taken",
    "catches": "catch taken",
    "caught out": "catch taken",
    "caught": "catch taken",
    "wicket taken": "wicket taken",
    "wickets taken": "wicket taken",
    "wicket": "wicket taken",
    "wickets": "wicket taken",
    "bowled out": "wicket taken",
    "bowled": "wicket taken",
    "clean bowled": "wicket taken",
    "lbw given": "LBW given",
    "lbws given": "LBW given",
    "lbw": "LBW given",
    "lbws": "LBW given",
    "leg before wicket": "LBW given",
    "leg before": "LBW given",
    "wicket-related": "dismissal",
    "wicket related": "dismissal",
    "wicket incidents": "dismissal",
    "wicket-taking": "dismissal",
    "dismissal": "dismissal",
    "dismissals": "dismissal",
    "dismissed": "dismissal",
    "outs": "dismissal",
    "out": "dismissal",

    # Boundaries
    "boundary hit": "boundary hit",
    "boundaries hit": "boundary hit",
    "boundary": "boundary hit",
    "boundaries": "boundary hit",
    "four": "boundary hit",
    "fours": "boundary hit",
    "4s": "boundary hit",
    "six hit": "six hit",
    "sixes hit": "six hit",
    "six": "six hit",
    "sixes": "six hit",
    "sixer": "six hit",
    "sixers": "six hit",
    "maximum": "six hit",
    "maximums": "six hit",
    "6s": "six hit",

    # Regular runs and over flow
    "run scored": "run scored",
    "runs scored": "run scored",
    "single": "run scored",
    "singles": "run scored",
    "over-completed": "over completed",
    "over-complete": "over completed",
    "over completed": "over completed",
    "overs completed": "over completed",
    "over completion": "over completed",
    "maiden": "over completed",
}

# Status Normalization Mapping (sort longer phrases first)
CANONICAL_STATUS_KEYWORDS: Dict[str, str] = {
    "awaiting verification": "pending review",
    "awaiting review": "pending review",
    "pending verification": "pending review",
    "yet to be verified": "pending review",
    "not yet verified": "pending review",
    "not verified": "pending review",
    "unverified": "pending review",
    "pending review": "pending review",
    "under review": "pending review",
    "in review": "pending review",
    "pending": "pending review",
    "disputed": "disputed",
    "dispute": "disputed",
    "controversial": "disputed",
    "controversy": "disputed",
    "confirmed": "confirmed",
    "confirm": "confirmed",
    "verified": "verified",
    "verify": "verified",
}

RE_VENUE = re.compile(r"\b(Stadium\s+[1-4])\b", re.IGNORECASE)
RE_WEATHER = re.compile(r"\b(sunny|overcast|humid|rainy|windy)\b", re.IGNORECASE)


def normalize_match_id(text: str) -> Optional[str]:
    """
    Extracts and standardizes match identifiers from diverse phrasing:
    'MATCH001', 'match 1', 'm1', 'game 1', '1st match', 'first match' -> 'MATCH001'.
    """
    if not text:
        return None

    # Check exact MATCH001
    exact_match = RE_MATCH_EXACT.search(text)
    if exact_match:
        return exact_match.group(1).upper()

    # Check numbered match, game, m
    numbered_match = RE_MATCH_NUMBERED.search(text)
    if numbered_match:
        num = int(numbered_match.group(1))
        return f"MATCH{num:03d}"

    # Check ordinal word e.g. "first match", "2nd game"
    ordinal_match = RE_MATCH_ORDINAL_WORD.search(text)
    if ordinal_match:
        word = ordinal_match.group(1).lower()
        if word in ORDINAL_MAP:
            return f"MATCH{ORDINAL_MAP[word]:03d}"

    return None


def normalize_team(text: str) -> Optional[str]:
    """
    Extracts and standardizes primary team identifier:
    'Team A', 'team-a', 'teama', 'side a' -> 'Team A'.
    """
    if not text:
        return None

    team_match = RE_TEAM.search(text)
    if team_match:
        letter = (team_match.group(1) or team_match.group(2)).upper()
        return f"Team {letter}"

    return None


def normalize_teams(text: str) -> List[str]:
    """
    Extracts all distinct team identifiers when multiple are mentioned:
    'Team A and Team B' -> ['Team A', 'Team B'].
    """
    if not text:
        return []

    found_teams = []
    for match in RE_TEAM.finditer(text):
        letter = (match.group(1) or match.group(2)).upper()
        t_name = f"Team {letter}"
        if t_name not in found_teams:
            found_teams.append(t_name)
    return found_teams


def normalize_over_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Detects semantic over phase ranges and explicit numerical ranges:
    - 'between overs 35 and 45' -> (35, 45)
    - 'first five overs', 'first 5 overs' -> (1, 5)
    - 'final ten overs', 'last 10 overs' -> (41, 50)
    - 'later overs', 'late overs', 'death overs' -> (30, 50)
    - 'early overs', 'powerplay' -> (1, 10)
    - 'middle overs' -> (11, 35)
    """
    if not text:
        return None, None

    lower = text.lower()

    # Explicit numerical range: 'between overs? X and Y' or 'overs? X to Y'
    m_between = re.search(r"between\s+overs?\s+(\d+)\s+(?:and|to)\s+(\d+)", lower)
    if m_between:
        return int(m_between.group(1)), int(m_between.group(2))

    m_range = re.search(r"overs?\s+(\d+)\s*(?:to|-)\s*(\d+)", lower)
    if m_range:
        return int(m_range.group(1)), int(m_range.group(2))

    # Number words mapping
    num_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "fifteen": 15, "twenty": 20,
    }

    # "first N overs"
    m_first = re.search(r"\b(?:first|initial)\s+(\w+|\d+)\s+overs?\b", lower)
    if m_first:
        val = m_first.group(1)
        n = int(val) if val.isdigit() else num_words.get(val)
        if n:
            return 1, n

    # "final N overs" or "last N overs"
    m_last = re.search(r"\b(?:final|last)\s+(\w+|\d+)\s+overs?\b", lower)
    if m_last:
        val = m_last.group(1)
        n = int(val) if val.isdigit() else num_words.get(val)
        if n:
            return max(1, 50 - n + 1), 50

    if any(p in lower for p in ["later overs", "late overs", "death overs", "final overs", "end of innings"]):
        return 30, 50
    if any(p in lower for p in ["early overs", "initial overs", "powerplay", "beginning of innings"]):
        return 1, 10
    if any(p in lower for p in ["middle overs"]):
        return 11, 35

    return None, None


def normalize_venue(text: str) -> Optional[str]:
    """Extracts stadium identifiers: 'Stadium 1', 'Stadium 2', 'Stadium 3', 'Stadium 4'."""
    if not text:
        return None
    m = RE_VENUE.search(text)
    return m.group(1).title() if m else None


def normalize_weather(text: str) -> Optional[str]:
    """Extracts weather conditions: 'sunny', 'overcast', 'humid', 'rainy'."""
    if not text:
        return None
    m = RE_WEATHER.search(text)
    return m.group(1).lower() if m else None


def normalize_player(text: str) -> Optional[str]:
    """
    Extracts and standardizes player identifiers:
    'Player 2', 'player-2', 'p2', 'p 2', 'batsman 2', 'bowler 2' -> 'Player 2'.
    """
    if not text:
        return None

    player_match = RE_PLAYER.search(text)
    if player_match:
        p_num = int(player_match.group(1) or player_match.group(2))
        return f"Player {p_num}"

    return None


def normalize_over(text: str) -> Optional[int]:
    """
    Extracts over numbers:
    'over 15', '15th over', 'ov 15' -> 15.
    """
    if not text:
        return None

    over_match = RE_OVER_NUMBERED.search(text)
    if over_match:
        return int(over_match.group(1))

    ord_match = RE_OVER_ORDINAL.search(text)
    if ord_match:
        return int(ord_match.group(1))

    return None


def normalize_event(text: str) -> Optional[str]:
    """
    Extracts cricket event from domain ontology dictionary.
    Prioritizes longer multi-word phrases over single keywords.
    """
    if not text:
        return None

    lower_text = text.lower()
    for kw, canonical_event in CANONICAL_EVENT_KEYWORDS.items():
        # Match as whole token or sub-phrase
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, lower_text):
            return canonical_event

    return None


def normalize_status(text: str) -> Optional[str]:
    """Extracts status flag using canonical mapping."""
    if not text:
        return None

    lower_text = text.lower()
    for kw, canonical_status in CANONICAL_STATUS_KEYWORDS.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, lower_text):
            return canonical_status

    return None


def normalize_statuses(text: str) -> List[str]:
    """Extracts all distinct status flags mentioned in text."""
    if not text:
        return []
    lower = text.lower()
    found = []
    if any(p in lower for p in ["awaiting verification", "awaiting review", "pending verification", "pending review", "unverified", "yet to be verified", "pending"]):
        found.append("pending review")
    if any(p in lower for p in ["disputed", "dispute", "controversial", "controversy"]):
        found.append("disputed")
    if any(p in lower for p in ["confirmed", "confirm"]):
        found.append("confirmed")
    if any(p in lower for p in ["verified", "verify"]):
        found.append("verified")
    return found


def normalize_role(text: str) -> Optional[str]:
    """Extracts player role: 'bowler', 'batsman', 'all-rounder', 'fielder'."""
    if not text:
        return None
    lower = text.lower()
    if re.search(r"\b(?:bowler|bowlers)\b", lower):
        return "bowler"
    if re.search(r"\b(?:batsman|batsmen|batter|batters)\b", lower):
        return "batsman"
    if re.search(r"\b(?:all-rounder|allrounder|all rounder)\b", lower):
        return "all-rounder"
    if re.search(r"\b(?:fielder|fielders)\b", lower):
        return "fielder"
    return None


def check_missing_venue(text: str) -> bool:
    """Checks if query asks for records where stadium/venue is omitted/missing."""
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in [
        "doesn't specify the stadium",
        "does not specify the stadium",
        "without specifying the stadium",
        "stadium is missing",
        "missing stadium",
        "stadium not specified",
        "venue not specified",
    ])


def check_missing_role(text: str) -> bool:
    """Checks if query asks for records where player role is omitted/missing."""
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in [
        "role is missing",
        "missing role",
        "without a role",
        "role not specified",
        "no player role",
        "missing from the commentary",
    ])


def extract_all_entities(query: str) -> Dict[str, Any]:
    """
    Master extractor resolving all cricket dimensions from any natural language query.
    """
    min_over, max_over = normalize_over_range(query)
    teams = normalize_teams(query)
    primary_team = teams[0] if teams else None
    statuses = normalize_statuses(query)
    primary_status = statuses[0] if len(statuses) == 1 else normalize_status(query)

    return {
        "match_id": normalize_match_id(query),
        "team": primary_team,
        "teams": teams,
        "player": normalize_player(query),
        "role": normalize_role(query),
        "over": normalize_over(query),
        "min_over": min_over,
        "max_over": max_over,
        "event": normalize_event(query),
        "status": primary_status,
        "statuses": statuses,
        "venue": normalize_venue(query),
        "weather": normalize_weather(query),
        "missing_venue": check_missing_venue(query),
        "missing_role": check_missing_role(query),
    }
