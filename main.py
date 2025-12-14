# app_enhanced_refactored.py – Sparky with inheritance-based agent architecture
from __future__ import annotations
from collections import Counter

import os, json, re, yaml
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Any, Tuple, Sequence, Annotated,Literal
from abc import ABC, abstractmethod

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import random
#from langchain_core.pydantic_v1 import BaseModel as PydanticV1BaseModel, Field as PydanticV1Field


try:
    from langchain_core.pydantic_v1 import BaseModel as PydanticV1BaseModel, Field as PydanticV1Field
    print("✅ Successfully imported from langchain_core.pydantic_v1")
except ImportError as e:
    print(f"❌ Failed to import from langchain_core.pydantic_v1: {e}")
    from pydantic.v1 import BaseModel as PydanticV1BaseModel, Field as PydanticV1Field
    print("✅ Successfully imported from pydantic.v1")

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

DEFAULT_CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", "try.yaml")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY and ChatOpenAI)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# Load agent + supervisor configuration from YAML
CONFIG_PATH = Path(DEFAULT_CONFIG_PATH)

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _raw_cfg = yaml.safe_load(f) or {}
else:
    _raw_cfg = {}
    print(f"WARNING: Config file not found at {CONFIG_PATH}. Using empty config.")

AGENT_CONFIGS: Dict[str, Dict[str, Any]] = _raw_cfg.get("agents", {})
SUPERVISOR_PROMPTS: Dict[str, str] = _raw_cfg.get("supervisor", {})

def format_affirmation(affirmation: str) -> str:
    """Format affirmation text in red and bold for Streamlit markdown."""
    if affirmation:
        return f"**<span style='color:red'>{affirmation}</span>**"
    return ""

def format_hook_text(hook_text: str) -> str:
    """Format hook text with larger font size for Streamlit markdown."""
    if hook_text:
        return f"<span style='font-size:30px'>{hook_text}</span>"
    return ""

# ============================================================================
# AD DATA INTEGRATION
# ============================================================================



AdTheme = Literal["self_expression", "wellness", "skill_growth", "ambition", "belonging"]

class AdData(TypedDict):
    ad_name: str
    ad_description: str
    ad_theme: AdTheme

MOCK_ADS: Dict[str, AdData] = {
    "extreme_talent": {
        "ad_name": "Talent Showcase",
        "ad_description": (
            "This ad uses impressive, high-skill, visually striking craft shots to trigger awe "
            "and a desire for mastery. It attracts Quiet Achievers and creative dabblers who want "
            "to feel more capable, skilled, and creatively confident. The message is about "
            "transforming from 'not good at this' into someone who produces beautiful, impressive "
            "work. The ad sells variety and mastery potential across many skills, creating a sense "
            "of progress, challenge, and personal capability — ideal for an achievement-driven funnel."
        ),
        "ad_theme": "confidence_progress",
    },
    "adhd_advantage": {
        "ad_name": "Neurodivergent Empowerment",
        "ad_description": (
            "This ad validates ADHD struggles with humor, relatability, and rapid-cut captions, "
            "promising a shift from shame and inconsistency to embracing ADHD as a creative advantage. "
            "It draws neurodivergent learners seeking self-acceptance and accessible learning structures. "
            "The emotional driver is relief + empowerment, not productivity, and it reassures them that "
            "Skill-A-Week works with their brain through flexibility, beginner-friendliness, and small wins."
        ),
        "ad_theme": "calm_wellbeing",
    },
    "cookies_asmr": {
        "ad_name": "Sensory Soother",
        "ad_description": (
            "This ad taps into sensory satisfaction and calming ASMR visuals, appealing to viewers seeking "
            "emotional relief, comfort, and a peaceful creative ritual. It attracts overwhelmed adults and "
            "sensory-sensitive people who want low-pressure, feel-good activities that help them unwind. "
            "The promise isn’t mastery — it’s gentle creativity and emotional soothing, framed as self-care."
        ),
        "ad_theme": "calm_wellbeing",
    },
    "purpose_tiktok": {
        "ad_name": "Creative Purpose",
        "ad_description": (
            "A reflective, emotionally resonant ad that speaks to people feeling directionless, uninspired, "
            "or disconnected from themselves. It offers creativity as a path to meaning, identity renewal, "
            "and inner clarity. It attracts seekers craving depth, intention, and personal rebirth, and the "
            "journey is about rediscovery rather than rushing toward skills or outcomes."
        ),
        "ad_theme": "exploration_discovery",
    },
    "narrator_and_lily": {
        "ad_name": "Heartfelt Connection",
        "ad_description": (
            "A warm, family-oriented storytelling ad centered on connection, presence, and shared creative "
            "moments. It appeals to parents, sentimental adults, and nostalgic creatives who value bonding "
            "and meaningful experiences. Creativity is framed as a way to nurture relationships and build memories."
        ),
        "ad_theme": "enrichment_purpose",
    },
    "money_making": {
        "ad_name": "Creative Income Builder",
        "ad_description": (
            "This ad targets financially motivated viewers seeking side income, financial relief, or "
            "entrepreneurial creative opportunities. It sells the transformation from feeling financially "
            "stuck to earning through creative output. The emotional driver is empowerment via financial "
            "autonomy, with emphasis on ROI, sellable skills, and practical steps to turn skills into income."
        ),
        "ad_theme": "confidence_progress",
    },
}
# ---------------------------------------------------------------------------
# Ensure connection_tone block exists with default emo fields
# --------------------------------------------------------------
#
def ensure_act_2_block(state: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee that stage_meta.act_2 exists with default emo fields."""
    stage_meta = dict(state.get("stage_meta", {}) or {})

    act_2_block = dict(stage_meta.get("act_2", {}) or {})
    act_2_meta = dict(act_2_block.get("metadata", {}) or {})
    act_2_state = dict(act_2_block.get("state", {}) or {})

    if "emo_act_2_type" not in act_2_meta:
        act_2_meta["emo_act_2_type"] = ""  # not decided yet
    if "confirm_act_2" not in act_2_meta:
        act_2_meta["confirm_act_2"] = "unclear"  # tone not confirmed yet
    if "act_2_emo_tone" not in act_2_meta:
        act_2_meta["act_2_emo_tone"] = "unclear"  # classification not set yet
    if "behavioral_signal" not in act_2_meta:
        act_2_meta["behavioral_signal"] = "mixed"  # safe neutral default

    # These two will get overwritten by the tone agent when it copies intent,
    # but we initialize them so the structure is always present.
    act_2_meta.setdefault("act_1_type", "")
    act_2_meta.setdefault("confirm_act_1", "unclear")

    # --- State defaults ---
    act_2_state.setdefault("last_theme", "")
    act_2_state.setdefault("last_level", "")

    # Add new emotional tone tracking fields
    act_2_state.setdefault("act_2_emo_1", "")
    act_2_state.setdefault("act_2_emo_2", "")
    act_2_state.setdefault("act_2_emo_3", "")

    # NEW: Add fields for flexible response formats
    act_2_state.setdefault("response_format_1", "multiple_choice")  # tracks format used for question 1
    act_2_state.setdefault("response_format_2", "multiple_choice")  # tracks format used for question 2
    act_2_state.setdefault("scale_range", "")  # e.g., "1-5" or "1-10" when using scales

    # turn = 0 means: tone agent has not asked any questions yet
    if "turn" not in act_2_state:
        act_2_state["turn"] = 0

    act_2_block["metadata"] = act_2_meta
    act_2_block["state"] = act_2_state
    stage_meta["act_2"] = act_2_block
    return {
        **state,
        "stage_meta": stage_meta,
    }


def get_mock_ad_data(ad_id: Optional[str] = None) -> AdData:
    """
    Mock ad data - replace this with actual external data source.

    You can optionally pass ad_id to choose a specific mock ad:
    - "extreme_talent"
    - "adhd_advantage"
    - "cookies_asmr"
    - "purpose_tiktok"
    - "narrator_and_lily"
    - "money_making"
    """

    # 1) If explicit ad_id was provided and exists, use it
    if ad_id and ad_id in MOCK_ADS:
        return MOCK_ADS[ad_id]

    # 2) Or use an env var (handy for testing without code changes)
    env_ad_id = os.getenv("MOCK_AD_ID")
    if env_ad_id and env_ad_id in MOCK_ADS:
        return MOCK_ADS[env_ad_id]

    # 3) Fallback: default scenario
    mock_ids = [
        "extreme_talent",
        "adhd_advantage",
        "cookies_asmr",
        "purpose_tiktok",
        "narrator_and_lily",
        "money_making",
    ]
    chosen_id = random.choice(mock_ids)
    return MOCK_ADS[chosen_id]

def get_ad_data_from_external_source(ad_id: Optional[str] = None) -> AdData:
    """Fetch ad data from external source (API, database, etc.).
    For now returns mock data, but structured to be easily replaced."""
    # TODO: Replace with actual external data fetching
    return get_mock_ad_data(ad_id=ad_id)


# ============================================================================
# STRUCTURED OUTPUTS
# ============================================================================

class AgentResponse(PydanticV1BaseModel):
    """Generic structured response used by ALL agents.

    - affirmation: warm reflection/validation (for connection_intent agent).
    - question_text: conversational question or prompt.
    - options: list of answer choices.
    - option_mapping: category mappings for each option (agent-specific categories).
    - metadata: agent-specific output (intent, act_3_type, barriers, persona, etc.).
    - state: agent-specific state (turn counts, levels, flags, progress).
    """

    affirmation: str = PydanticV1Field(
        default="",
        description="Warm reflection/validation (used by connection_intent agent)"
    )

    question_text: str = PydanticV1Field(
        default="",
        description="The conversational question or prompt to show the user"
    )

    options: List[str] = PydanticV1Field(
        default_factory=list,
        description="Answer options (any number, typically 4)"
    )

    option_mapping: List[str] = PydanticV1Field(
        ...,  # Required field - NO default!
        description=(
            "Category mapping for each option (4 items). "
            "Categories are agent-specific: intent categories for Act 1, emotional categories for Act 2, etc. "
            "Example: ['belonging', 'skill_growth', 'wellness', 'unsure']"
        ),
        min_items=4,
        max_items=4
    )

    metadata: Dict[str, Any] = PydanticV1Field(
        default_factory=dict,
        description="Agent-specific metadata for routing or inference",
    )

    state: Dict[str, Any] = PydanticV1Field(
        default_factory=dict,
        description="Agent-specific internal state (e.g., turn numbers, progress flags, counters)",
    )

    class Config:
        extra = "forbid"


# Strict response model specifically for connection_intent agent
class ConnectionIntentResponse(PydanticV1BaseModel):
    """Strict response format for connection_intent agent only."""

    affirmation: str = PydanticV1Field(
        ...,  # Required field (no default)
        description="Warm reflection/validation sentence (1-2 sentences)",
        min_length=10
    )

    question_text: str = PydanticV1Field(
        ...,  # Required field
        description="First-person question to ask the user",
        min_length=10
    )

    options: List[str] = PydanticV1Field(
        ...,  # Required field
        description="Exactly 4 answer options",
        min_items=4,
        max_items=4
    )

    option_mapping: List[str] = PydanticV1Field(  # ← Renamed from act_1_mapping
        ...,  # Required field - NO default!
        description=(
            "Exactly 4 intent categories matching the 4 options. "
            "Each value must be one of: 'self_expression', 'wellness', 'skill_growth', 'ambition', 'belonging', 'unsure'. "
            "Example: ['belonging', 'skill_growth', 'wellness', 'unsure']"
        ),
        min_items=4,
        max_items=4
    )

    class Config:
        extra = "forbid"


# Now add EmotionalToneResponse right after this
class EmotionalToneResponse(PydanticV1BaseModel):
    """Strict response format for emotional_tone agent only.
    Supports multiple response formats: multiple_choice, yes_no, and scale."""

    affirmation: str = PydanticV1Field(
        ...,  # Required field (no default)
        description="Warm, caring reflection sentence (1-2 sentences)",
        min_length=10
    )

    question_text: str = PydanticV1Field(
        ...,  # Required field
        description="First-person question about emotional feelings",
        min_length=10
    )

    response_format: str = PydanticV1Field(
        default="multiple_choice",
        description="Type of response expected: 'multiple_choice', 'yes_no', or 'scale'"
    )

    # Fields for multiple_choice and yes_no formats
    options: Optional[List[str]] = PydanticV1Field(
        default=None,
        description="Answer options (4 for multiple_choice, 2 for yes_no, None for scale)"
    )

    act_2_emotional_mapping: Optional[List[str]] = PydanticV1Field(
        default=None,
        description=(
            "Emotional tone categories matching the options. "
            "Values: 'positive', 'neutral', 'tense', 'resistant', 'unsure', 'mixed'. "
            "Required for multiple_choice and yes_no, None for scale."
        )
    )

    # Fields for scale format
    scale_range: Optional[str] = PydanticV1Field(
        default=None,
        description="Scale range like '1-5' or '1-10' (only for scale format)"
    )

    scale_labels: Optional[Dict[str, str]] = PydanticV1Field(
        default=None,
        description="Optional labels for scale endpoints, e.g., {'min': 'Not at all', 'max': 'Extremely'}"
    )
    scale_mapping: Optional[Dict[str, str]] = PydanticV1Field(
        default=None,
        description = "Mapping of number ranges to emotional categories. Keys are ranges like '1-3', '4-7', values are categories like 'neutral', 'tense'. Only for scale format."
    )

    class Config:
        extra = "forbid"
# Hook Response for hook agent
class HookResponse(PydanticV1BaseModel):
    hook_text: str = PydanticV1Field(description="The 2-3 sentence hook message")

    class Config:
        extra = "forbid"

# Motivation Response for motivation agent

class MotivationResponse(PydanticV1BaseModel):
    """Strict response format for motivation agent only."""

    affirmation: str = PydanticV1Field(
        ...,  # Required field (no default)
        description="Warm reflection/validation sentence (1-2 sentences)",
        min_length=10
    )

    question_text: str = PydanticV1Field(
        ...,  # Required field
        description="First-person question to ask the user about their motivation",
        min_length=10
    )

    options: List[str] = PydanticV1Field(
        ...,  # Required field
        description="Exactly 4 answer options",
        min_items=4,
        max_items=4
    )

    act_3_mapping: List[str] = PydanticV1Field(
        ...,  # Required field - NO default!
        description=(
            "Exactly 4 motivation categories matching the 4 options. "
            "Each value must be one of: 'progress', 'identity', 'relief', 'play', 'belonging', 'achievement', 'autonomy', 'expression', 'unsure'. "
            "Example: ['progress', 'achievement', 'belonging', 'unsure']"
        ),
        min_items=4,
        max_items=4
    )

    class Config:
        extra = "forbid"
# Barriers Response for barriers agent
class BarriersResponse(PydanticV1BaseModel):
    """Strict response format for barriers agent only."""

    affirmation: str = PydanticV1Field(
        ...,  # Required field (no default)
        description="Warm reflection/validation sentence (1-2 sentences)",
        min_length=10
    )

    question_text: str = PydanticV1Field(
        ...,  # Required field
        description="First-person question to ask the user about their barriers",
        min_length=10
    )

    options: List[str] = PydanticV1Field(
        ...,  # Required field
        description="Exactly 4 answer options",
        min_items=4,
        max_items=4
    )

    act_4_mapping: List[str] = PydanticV1Field(
        ...,  # Required field - NO default!
        description=(
            "Exactly 4 barrier categories matching the 4 options. "
            "Each value must be one of: 'consistency', 'overwhelm', 'time_energy', 'confidence', "
            "'clarity', 'perfectionism', 'motivation_drop', 'fear_block', 'structure', 'distraction', 'unsure'. "
            "Example: ['consistency', 'overwhelm', 'clarity', 'unsure']"
        ),
        min_items=4,
        max_items=4
    )

    class Config:
        extra = "forbid"
# Summary Response for summary agent
class SummaryResponse(PydanticV1BaseModel):
    """Strict response format for summary agent only."""

    summary_text: str = PydanticV1Field(
        ...,  # Required field (no default)
        description="Warm, personalized summary of user's journey (50-150 words)",
        min_length=50
    )

    recommendations: List[str] = PydanticV1Field(
        ...,  # Required field
        description="2-3 actionable recommendations or next steps",
        min_items=2,
        max_items=3
    )

    class Config:
        extra = "forbid"

def get_user_chosen_act_1(user_input: str, options: List[str], option_mapping: List[str]) -> str:
    """
    Determine which intent the user chose based on their input.

    Args:
        user_input: The user's message (e.g., "A" or "Build new skills")
        options: List of option texts from previous turn
        option_mapping: List of intent categories matching each option

    Returns:
        The intent category (e.g., "skill_growth", "wellness", "unsure")
    """
    if not options or not option_mapping:
        return "unsure"

    user_input_clean = user_input.strip().lower()

    # Check if user typed A, B, C, D
    if len(user_input_clean) == 1 and user_input_clean in ['a', 'b', 'c', 'd']:
        idx = ord(user_input_clean) - ord('a')
        if 0 <= idx < len(option_mapping):
            return option_mapping[idx]

    # Check if user typed the full option text
    for i, option in enumerate(options):
        if option.lower() in user_input_clean or user_input_clean in option.lower():
            if i < len(option_mapping):
                return option_mapping[i]

    # Fallback
    return "unsure"


def compute_question_direction(theme_1: str, theme_2: str, current_turn: int) -> str:
    """
    Determine if next question should be 'broad' or 'focused'.

    Fixed pattern:
    - Turn 1: broad (aspiration)
    - Turn 2: focused (aspiration)
    - Turn 3: broad (identity shift)
    - Turn 4: focused (identity shift)
    """
    if current_turn == 1:
        return "broad"
    elif current_turn == 2:
        return "focused"
    elif current_turn == 3:
        return "broad"
    elif current_turn == 4:
        return "focused"
    else:
        return "broad"  # Fallback


def should_finalize_act_1(theme_1: str, theme_2: str, theme_3: str, theme_4: str, ad_theme: str) -> Tuple[bool, str]:
    """
    Finalize ONLY after all 4 intent questions have been answered.
    Majority vote:
      - If any two or more match → that theme
      - If all different → 'mixed'
    """
    from collections import Counter

    # --- CASE 1: Haven't reached question 4 yet ---
    if not theme_4 or theme_4 == "":
        return False, ""

    # --- CASE 2: We HAVE 4 answers, finalize ALWAYS ---
    themes = [theme_1, theme_2, theme_3, theme_4]

    # Filter out empty/unsure for counting
    valid_themes = [t for t in themes if t and t != "unsure"]

    if not valid_themes:
        return True, "unsure"

    # Count occurrences
    counts = Counter(valid_themes)
    most_common = counts.most_common(1)[0]

    # If we have a clear majority (appears 2+ times)
    if most_common[1] >= 2:
        return True, most_common[0]

    # All different → mixed
    return True, "mixed"


def get_user_chosen_act_2(user_input: str, options: List[str], act_2_emotional_mapping: List[str]) -> str:
    """
    Determine which emotional tone the user chose based on their input.

    Args:
        user_input: The user's answer text
        options: List of 4 option texts
        act_2_emotional_mapping: List of 4 emotional tone categories corresponding to options

    Returns:
        The emotional tone category the user selected
    """
    if not act_2_emotional_mapping or len(act_2_emotional_mapping) != 4:
        print(f"WARNING - get_user_chosen_act_2: invalid act_2_emotional_mapping: {act_2_emotional_mapping}")
        return "unsure"

    if not options or len(options) != 4:
        print(f"WARNING - get_user_chosen_act_2: invalid options: {options}")
        return "unsure"

    user_lower = user_input.lower().strip()

    # Try to match user input to one of the options
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        # Exact match or significant overlap
        if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
            return act_2_emotional_mapping[i]

    # Fallback: couldn't determine
    print(f"WARNING - get_user_chosen_act_2: couldn't match '{user_input}' to options")
    return "unsure"


def compute_final_act_2(act_2_emo_1: str, act_2_emo_2: str, act_2_emo_3: str, act_2_emo_4: str) -> str:
    """
    Determine the final emotional tone from 4 answers.

    Logic (similar to Act 1):
    1. If 2+ answers are "unsure" → "unsure"
    2. If same tone appears 2+ times → that tone
    3. If tie (2 tones with 2 votes each) → use most recent
    4. If all different → "mixed"

    Args:
        act_2_emo_1: Emotional tone from Q1
        act_2_emo_2: Emotional tone from Q2
        act_2_emo_3: Emotional tone from Q3
        act_2_emo_4: Emotional tone from Q4

    Returns:
        Final emotional tone category
    """
    from collections import Counter

    # Normalize inputs
    tones = [
        (act_2_emo_1 or "").lower().strip(),
        (act_2_emo_2 or "").lower().strip(),
        (act_2_emo_3 or "").lower().strip(),
        (act_2_emo_4 or "").lower().strip()
    ]

    # If any are missing, return unsure
    if not all(tones):
        return "unsure"

    # RULE 1: If 2+ answers are "unsure" → "unsure"
    unsure_count = sum(1 for t in tones if t == "unsure")
    if unsure_count >= 2:
        return "unsure"

    # Filter out empty and unsure
    filtered = [t for t in tones if t and t != "unsure"]

    if not filtered:
        return "unsure"

    # RULE 2: Count occurrences
    counts = Counter(filtered)
    max_count = max(counts.values())

    # If we have a clear winner (appears 2+ times)
    if max_count >= 2:
        candidates = [t for t, c in counts.items() if c == max_count]

        # Tiebreaker: prefer most recent
        # Priority order: act_2_emo_4 > act_2_emo_3 > act_2_emo_2 > act_2_emo_1
        priority_order = [tones[3], tones[2], tones[1], tones[0]]
        for p in priority_order:
            if p in candidates:
                return p

        # Fallback
        return candidates[0]

    # RULE 3: All different → "mixed"
    return "mixed"


def should_finalize_act_2(act_2_emo_1: str, act_2_emo_2: str, act_2_emo_3: str, act_2_emo_4: str) -> Tuple[bool, str]:
    """
    Determine if emotional tone assessment is complete and what the final tone is.

    Args:
        act_2_emo_1: User's emotional tone from Q1
        act_2_emo_2: User's emotional tone from Q2
        act_2_emo_3: User's emotional tone from Q3
        act_2_emo_4: User's emotional tone from Q4

    Returns:
        Tuple of (should_finalize, final_tone)
        - should_finalize: True if we have 4 answers
        - final_tone: The determined emotional tone
    """
    # Need all 2 answers to finalize
    if not act_2_emo_4 or act_2_emo_4 == "":  # ← CORRECT: act_2_emo_4
        return (False, "")

    # We have 2 answers, time to finalize
    # Call the logic function to determine final tone
    final_tone = compute_final_act_2(act_2_emo_1, act_2_emo_2, act_2_emo_3, act_2_emo_4)

    return (True, final_tone)

def update_act_1_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update intent metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored option_mapping
    2. Updates theme_1, theme_2, or theme_3 based on current turn
    3. Checks if intent should be finalized
    4. Updates confirm_act_1 and act_1_type
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    act_1_block = stage_meta.get("act_1", {}) or {}
    act_1_state = dict(act_1_block.get("state", {}) or {})
    act_1_meta = dict(act_1_block.get("metadata", {}) or {})

    # Get stored mapping from previous agent run
    prev_option_mapping = act_1_state.get("option_mapping", [])
    prev_options = act_1_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = act_1_state.get("turn", 0)

    # Only process if we have a mapping and the user actually answered
    if not prev_option_mapping or not user_answer:
        return state

    # Classify the user's answer
    chosen_act_1 = get_user_chosen_act_1(user_answer, prev_options, prev_option_mapping)

    print(
        f"DEBUG - update_act_1_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_act_1='{chosen_act_1}'")

    # Update theme based on which question they just answered
    if current_turn == 1:
        # They just answered Q1
        act_1_state["theme_1"] = chosen_act_1
        act_1_state["last_theme"] = chosen_act_1
    elif current_turn == 2:
        # They just answered Q2
        act_1_state["theme_2"] = chosen_act_1
        act_1_state["last_theme"] = chosen_act_1
    elif current_turn == 3:
        # They just answered Q3
        act_1_state["theme_3"] = chosen_act_1
        act_1_state["last_theme"] = chosen_act_1
    elif current_turn == 4:
        # They just answered Q4
        act_1_state["theme_4"] = chosen_act_1
        act_1_state["last_theme"] = chosen_act_1

    # Get all four themes
    theme_1 = act_1_state.get("theme_1", "")
    theme_2 = act_1_state.get("theme_2", "")
    theme_3 = act_1_state.get("theme_3", "")
    theme_4 = act_1_state.get("theme_4", "")

    # Get ad_theme for finalization logic
    ad_data = state.get("ad_data", {}) or {}
    ad_theme = ad_data.get("ad_theme", "")

    # Check if we should finalize (after 4 questions)
    should_finalize, act_1_type = should_finalize_act_1(theme_1, theme_2, theme_3, theme_4, ad_theme)

    print(f"DEBUG - should_finalize={should_finalize}, act_1_type='{act_1_type}'")

    if should_finalize:
        # We're done with intent questions - finalize
        act_1_meta["confirm_act_1"] = "clear"
        act_1_meta["act_1_type"] = act_1_type
    else:
        # Still asking questions - but update act_1_type progressively
        act_1_meta["confirm_act_1"] = "unclear"

        # PROGRESSIVE UPDATE: Set act_1_type based on what we know so far
        if current_turn >= 3 and theme_3:
            # After Q3 (identity question), set act_1_type to theme_3
            # This gives Act 2 the identity context immediately
            act_1_meta["act_1_type"] = theme_3
        elif current_turn >= 2 and theme_2:
            # After Q2 (focused aspiration), set to theme_2
            act_1_meta["act_1_type"] = theme_2
        elif current_turn >= 1 and theme_1:
            # After Q1 (broad aspiration), set to theme_1
            act_1_meta["act_1_type"] = theme_1
        # If no themes yet, leave act_1_type as is (don't clear it)

    # Write back the updated metadata and state
    stage_meta["act_1"] = {
        "metadata": act_1_meta,
        "state": act_1_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }


def update_act_2_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update emotional tone metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored act_2_emotional_mapping OR scale_mapping
    2. Updates act_2_emo_1 or act_2_emo_2 based on current turn
    3. Checks if emotional tone should be finalized
    4. Updates confirm_act_2 and act_2_emo_tone
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    act_2_block = stage_meta.get("act_2", {}) or {}
    act_2_state = dict(act_2_block.get("state", {}) or {})
    act_2_meta = dict(act_2_block.get("metadata", {}) or {})

    # Get current turn (which question did they just answer?)
    current_turn = act_2_state.get("turn", 0)

    if not user_answer or current_turn == 0:
        return state
    # Determine which format was used for this question
    if current_turn == 1:
        response_format = act_2_state.get("response_format_1", "multiple_choice")
    elif current_turn == 2:
        response_format = act_2_state.get("response_format_2", "multiple_choice")
    elif current_turn == 3:
        response_format = act_2_state.get("response_format_3", "multiple_choice")
    elif current_turn == 4:
        response_format = act_2_state.get("response_format_4", "multiple_choice")
    else:
        response_format = "multiple_choice"


    # Classify the user's answer based on format
    chosen_act_2 = ""

    if response_format == "scale":
        # For scale questions, use scale_mapping
        scale_mapping = act_2_state.get("scale_mapping", {})
        if scale_mapping and user_answer.isdigit():
            scale_value = int(user_answer)

            # Find which range this value falls into
            for range_str, category in scale_mapping.items():
                if "-" in range_str:
                    min_val, max_val = map(int, range_str.split("-"))
                    if min_val <= scale_value <= max_val:
                        chosen_act_2 = category
                        break
                elif range_str.isdigit() and int(range_str) == scale_value:
                    # Exact match (e.g., "3": "neutral")
                    chosen_act_2 = category
                    break
    else:
        # For multiple_choice and yes_no, use act_2_emotional_mapping
        prev_act_2_emotional_mapping = act_2_state.get("act_2_emotional_mapping", [])
        prev_options = act_2_state.get("options", [])

        # Filter out empty strings from padding
        prev_act_2_emotional_mapping = [m for m in prev_act_2_emotional_mapping if m]
        prev_options = [o for o in prev_options if o]

        if prev_act_2_emotional_mapping and prev_options:
            chosen_act_2 = get_user_chosen_act_2(user_answer, prev_options, prev_act_2_emotional_mapping)

    print(
        f"DEBUG - update_emotional_act_2_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_act_2='{chosen_act_2}'")

    # Update act_2_emo_tone based on which question they just answered
    if current_turn == 1:
        # They just answered Q1
        act_2_state["act_2_emo_1"] = chosen_act_2
        act_2_state["last_theme"] = chosen_act_2
    elif current_turn == 2:
        # They just answered Q2
        act_2_state["act_2_emo_2"] = chosen_act_2
        act_2_state["last_theme"] = chosen_act_2
    elif current_turn == 3:
        # They just answered Q3
        act_2_state["act_2_emo_3"] = chosen_act_2
        act_2_state["last_theme"] = chosen_act_2
    elif current_turn == 4:
        # They just answered Q4
        act_2_state["act_2_emo_4"] = chosen_act_2
        act_2_state["last_theme"] = chosen_act_2

    # Get all four tones
    act_2_emo_1 = act_2_state.get("act_2_emo_1", "")
    act_2_emo_2 = act_2_state.get("act_2_emo_2", "")
    act_2_emo_3 = act_2_state.get("act_2_emo_3", "")
    act_2_emo_4 = act_2_state.get("act_2_emo_4", "")

    # Check if we should finalize (after 4 questions)
    should_finalize, final_tone = should_finalize_act_2(act_2_emo_1, act_2_emo_2, act_2_emo_3, act_2_emo_4)

    print(f"DEBUG - should_finalize={should_finalize}, final_tone='{final_tone}'")

    if should_finalize:
        # We're done with emotional tone questions
        act_2_meta["confirm_act_2"] = "clear"
        act_2_meta["act_2_emo_tone"] = final_tone
    else:
        # Keep asking questions
        act_2_meta["confirm_act_2"] = "unclear"
        act_2_meta["act_2_emo_tone"] = "unclear"

    # Write back the updated metadata and state
    stage_meta["act_2"] = {
        "metadata": act_2_meta,
        "state": act_2_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }
def get_user_chosen_act_3(user_input: str, options: List[str], act_3_mapping: List[str]) -> str:
    """
    Determine which motivation the user chose based on their input.

    Args:
        user_input: The user's answer text
        options: List of 4 option texts
        act_3_mapping: List of 4 motivation categories corresponding to options

    Returns:
        The motivation category the user selected
    """
    if not act_3_mapping or len(act_3_mapping) != 4:
        print(f"WARNING - get_user_chosen_act_3: invalid act_3_mapping: {act_3_mapping}")
        return "unsure"

    if not options or len(options) != 4:
        print(f"WARNING - get_user_chosen_act_3: invalid options: {options}")
        return "unsure"

    user_lower = user_input.lower().strip()

    # Try to match user input to one of the options
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        # Exact match or significant overlap
        if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
            return act_3_mapping[i]

    # Fallback: couldn't determine
    print(f"WARNING - get_user_chosen_act_3: couldn't match '{user_input}' to options")
    return "unsure"


def compute_question_direction_act_3(act_3_answer_1: str, act_3_answer_2: str, act_3_answer_3: str,
                                          current_turn: int) -> str:
    """
    Determine if next motivation question should be 'broad' or 'deep'.

    Logic (same as intent agent):
    - Turn 1: Always broad (exploring from intent)
    - Turn 2: Deep if act_3_answer_1 is clear and not unsure, broad otherwise
    - Turn 3: Deep if act_3_answer_1 == act_3_answer_2 (consistent), broad if mixed or unsure
    - Turn 4: Deep if consistent pattern, broad if mixed

    Args:
        act_3_answer_1: User's motivation from Q1
        act_3_answer_2: User's motivation from Q2
        act_3_answer_3: User's motivation from Q3
        current_turn: The question number we're about to ask (1-4)

    Returns:
        "broad" or "deep"
    """
    if current_turn == 1:
        return "broad"

    if current_turn == 2:
        # We have act_3_answer_1 now
        if act_3_answer_1 in ["unsure", ""]:
            return "broad"
        else:
            return "deep"  # User showed clear motivation in Q1

    if current_turn == 3:
        # We have act_3_answer_1 and act_3_answer_2
        if act_3_answer_1 == act_3_answer_2 and act_3_answer_1 not in ["unsure", ""]:
            return "deep"  # Consistent motivation
        else:
            return "broad"  # Mixed or unsure

    if current_turn == 4:
        # We have act_3_answer_1, act_3_answer_2, and act_3_answer_3
        # Check for consistency
        motivations = [act_3_answer_1, act_3_answer_2, act_3_answer_3]
        # Remove unsure/empty
        clear_motivations = [m for m in motivations if m and m != "unsure"]

        if len(clear_motivations) >= 2:
            # Check if at least 2 are the same
            if clear_motivations[0] == clear_motivations[-1] or (
                    len(clear_motivations) >= 2 and clear_motivations[0] == clear_motivations[1]):
                return "deep"

        return "broad"

    return "broad"  # Fallback

def compute_question_direction_act_2(current_turn: int) -> str:
    """
    Determine if next question should be 'broad' or 'focused' for Act 2.

    Fixed pattern:
    - Turn 1: broad (learning pattern)
    - Turn 2: focused (learning pattern)
    - Turn 3: broad (engagement pattern)
    - Turn 4: focused (engagement pattern)
    """
    if current_turn == 1:
        return "broad"
    elif current_turn == 2:
        return "focused"
    elif current_turn == 3:
        return "broad"
    elif current_turn == 4:
        return "focused"
    else:
        return "broad"  # Fallback
def compute_final_motivation(act_3_answer_1: str, act_3_answer_2: str, act_3_answer_3: str, act_3_answer_4: str,
                             act_1_type: str) -> str:
    """
    Determine the final motivation from 4 answers.

    Logic:
    1. If 2+ answers are "unsure" → "unsure"
    2. If same motivation appears 2+ times → that motivation
    3. If tie (2 motivations with 2 votes each) → use most recent
    4. If all different → "mixed"

    Args:
        act_3_answer_1: Motivation from Q1
        act_3_answer_2: Motivation from Q2
        act_3_answer_3: Motivation from Q3
        act_3_answer_4: Motivation from Q4
        act_1_type: User's intent (used as tiebreaker if needed)

    Returns:
        Final motivation category
    """
    # Normalize inputs
    motivations = [
        (act_3_answer_1 or "").lower().strip(),
        (act_3_answer_2 or "").lower().strip(),
        (act_3_answer_3 or "").lower().strip(),
        (act_3_answer_4 or "").lower().strip()
    ]

    # If any are missing, return unsure
    if not all(motivations):
        return "unsure"

    # RULE 1: If 2+ answers are "unsure" → "unsure"
    unsure_count = sum(1 for m in motivations if m == "unsure")
    if unsure_count >= 2:
        return "unsure"

    # RULE 2: Count occurrences (similar to intent logic)
    from collections import Counter

    # Include act_1_type as a tiebreaker option
    all_values = motivations + [act_1_type.lower().strip()] if act_1_type else motivations

    # Filter out empty and unsure
    filtered = [m for m in motivations if m and m != "unsure"]

    if not filtered:
        return "unsure"

    counts = Counter(filtered)
    max_count = max(counts.values())

    # If we have a clear winner (appears 2+ times)
    if max_count >= 2:
        candidates = [m for m, c in counts.items() if c == max_count]

        # Tiebreaker: prefer most recent
        # Priority order: act_3_answer_4 > act_3_answer_3 > act_3_answer_2 > act_3_answer_1
        priority_order = [motivations[3], motivations[2], motivations[1], motivations[0]]
        for p in priority_order:
            if p in candidates:
                return p

        # Fallback
        return candidates[0]

    # RULE 3: All different → "mixed"
    return "mixed"
def should_finalize_act_3(act_3_answer_1: str, act_3_answer_2: str, act_3_answer_3: str, act_3_answer_4: str, act_1_type: str) -> Tuple[bool, str]:
    """
    Determine if motivation assessment is complete and what the final motivation is.

    Args:
        act_3_answer_1: User's motivation from Q1
        act_3_answer_2: User's motivation from Q2
        act_3_answer_3: User's motivation from Q3
        act_3_answer_4: User's motivation from Q4
        act_1_type: User's intent type (for tiebreaker)

    Returns:
        Tuple of (should_finalize, final_motivation)
        - should_finalize: True if we have 4 answers
        - final_motivation: The determined motivation type
    """
    # Need all 4 answers to finalize
    if not act_3_answer_4 or act_3_answer_4 == "":
        return (False, "")

    # We have 4 answers, time to finalize
    # Call the logic function to determine final motivation
    final_motivation = compute_final_motivation(act_3_answer_1, act_3_answer_2, act_3_answer_3, act_3_answer_4, act_1_type)

    return (True, final_motivation)

def update_act_3_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update motivation metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored act_3_mapping
    2. Updates act_3_answer_1, act_3_answer_2, act_3_answer_3, or act_3_answer_4 based on current turn
    3. Updates last_act_3 to most recent answer
    4. Checks if motivation should be finalized
    5. Updates confirm_act_3 and act_3_type
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    act_3_block = stage_meta.get("act_3", {}) or {}
    act_3_state = dict(act_3_block.get("state", {}) or {})
    act_3_meta = dict(act_3_block.get("metadata", {}) or {})

    # Get stored mapping from previous agent run
    prev_act_3_mapping = act_3_state.get("act_3_mapping", [])
    prev_options = act_3_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = act_3_state.get("turn", 0)

    # Only process if we have a mapping and the user actually answered
    if not prev_act_3_mapping or not user_answer:
        return state

    # Classify the user's answer
    chosen_act_3 = get_user_chosen_act_3(user_answer, prev_options, prev_act_3_mapping)

    print(
        f"DEBUG - update_act_3_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_act_3='{chosen_act_3}'")

    # Update motivation based on which question they just answered
    # Update motivation based on which question they just answered
    if current_turn == 1:
        # They just answered Q1
        act_3_state["act_3_answer_1"] = chosen_act_3
        act_3_state["last_act_3"] = chosen_act_3
        act_3_state["last_theme"] = chosen_act_3
    elif current_turn == 2:
        # They just answered Q2
        act_3_state["act_3_answer_2"] = chosen_act_3
        act_3_state["last_act_3"] = chosen_act_3
        act_3_state["last_theme"] = chosen_act_3
    elif current_turn == 3:
        # They just answered Q3
        act_3_state["act_3_answer_3"] = chosen_act_3
        act_3_state["last_act_3"] = chosen_act_3
        act_3_state["last_theme"] = chosen_act_3
    elif current_turn == 4:
        # They just answered Q4
        act_3_state["act_3_answer_4"] = chosen_act_3
        act_3_state["last_act_3"] = chosen_act_3
        act_3_state["last_theme"] = chosen_act_3

    # Get all four motivations
    act_3_answer_1 = act_3_state.get("act_3_answer_1", "")
    act_3_answer_2 = act_3_state.get("act_3_answer_2", "")
    act_3_answer_3 = act_3_state.get("act_3_answer_3", "")
    act_3_answer_4 = act_3_state.get("act_3_answer_4", "")

    # Get act_1_type for finalization logic
    act_1_block = stage_meta.get("act_1", {}) or {}
    act_1_meta = act_1_block.get("metadata", {}) or {}
    act_1_type = act_1_meta.get("act_1_type", "")

    # Check if we should finalize (after 4 questions)
    should_finalize, final_motivation = should_finalize_act_3(
        act_3_answer_1, act_3_answer_2, act_3_answer_3, act_3_answer_4, act_1_type
    )

    print(f"DEBUG - should_finalize={should_finalize}, final_motivation='{final_motivation}'")

    if should_finalize:
        # We're done with motivation questions
        act_3_meta["confirm_act_3"] = "clear"
        act_3_meta["act_3_type"] = final_motivation
    else:
        # Keep asking questions
        act_3_meta["confirm_act_3"] = "unclear"
        act_3_meta["act_3_type"] = ""

    # Write back the updated metadata and state
    stage_meta["act_3"] = {
        "metadata": act_3_meta,
        "state": act_3_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }
def get_user_chosen_act_4(user_input: str, options: List[str], act_4_mapping: List[str]) -> str:
    """
    Determine which barrier the user chose based on their input.

    Args:
        user_input: The user's answer text
        options: List of 4 option texts
        act_4_mapping: List of 4 barrier categories corresponding to options

    Returns:
        The barrier category the user selected
    """
    if not act_4_mapping or len(act_4_mapping) != 4:
        print(f"WARNING - get_user_chosen_act_4: invalid act_4_mapping: {act_4_mapping}")
        return "unsure"

    if not options or len(options) != 4:
        print(f"WARNING - get_user_chosen_act_4: invalid options: {options}")
        return "unsure"

    user_lower = user_input.lower().strip()

    # Try to match user input to one of the options
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        # Exact match or significant overlap
        if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
            return act_4_mapping[i]

    # Fallback: couldn't determine
    print(f"WARNING - get_user_chosen_act_4: couldn't match '{user_input}' to options")
    return "unsure"
def compute_question_direction_act_4(act_4_answer_1: str,
                                        current_turn: int) -> str:
    """
    Determine if next barrier question should be 'broad' or 'focused'.

    Logic:
    - Turn 1: Always broad (exploring support needs)
    - Turn 2: Focused if act_4_answer_1 is clear and not unsure, broad otherwise

    Args:
        act_4_answer_1: User's support preference from Q1
        current_turn: The question number we're about to ask (1-2)

    Returns:
        "broad" or "focused"
    """
    if current_turn == 1:
        return "broad"

    if current_turn == 2:
        # We have act_4_answer_1 now
        if act_4_answer_1 in ["unsure", ""]:
            return "broad"
        else:
            return "focused"  # User showed clear support preference in Q1

    # Only 2 questions - default to broad as fallback
    return "broad"




def compute_final_barrier(act_4_answer_1: str, act_4_answer_2: str,
                         act_3_type: str) -> str:
    """
    Determine the final barrier from 2 answers.

    Logic:
    1. If 2+ answers are "unsure" → "unsure"
    2. If same barrier appears 2+ times → that barrier
    3. If tie (2 barriers with 2 votes each) → use most recent
    4. If all different → "mixed"

    Args:
        act_4_answer_1: Barrier from Q1
        act_4_answer_2: Barrier from Q2
        act_3_type: User's motivation (used as tiebreaker if needed)

    Returns:
        Final barrier category
    """
    # Normalize inputs
    barriers = [
        (act_4_answer_1 or "").lower().strip(),
        (act_4_answer_2 or "").lower().strip()
    ]

    # If any are missing, return unsure
    if not all(barriers):
        return "unsure"

    # RULE 1: If 2+ answers are "unsure" → "unsure"
    unsure_count = sum(1 for b in barriers if b == "unsure")
    if unsure_count >= 2:
        return "unsure"

    # RULE 2: Count occurrences (similar to intent/motivation logic)
    from collections import Counter

    # Filter out empty and unsure
    filtered = [b for b in barriers if b and b != "unsure"]

    if not filtered:
        return "unsure"

    counts = Counter(filtered)
    max_count = max(counts.values())

    # If we have a clear winner (appears 2+ times)
    if max_count >= 2:
        candidates = [b for b, c in counts.items() if c == max_count]

        # Tiebreaker: prefer most recent
        # Priority order: act_4_answer_2 > act_4_answer_1
        priority_order = [barriers[1], barriers[0]]
        for p in priority_order:
            if p in candidates:
                return p

        # Fallback
        return candidates[0]
    # RULE 3: Both different → use most recent (answer_2)
    if barriers[1]:
        return barriers[1]
    return barriers[0] if barriers[0] else "unsure"


def should_finalize_act_4(act_4_answer_1: str, act_4_answer_2: str, act_4_answer_3: str, act_4_answer_4: str,
                            act_3_type: str) -> Tuple[bool, str]:
    """
    Determine if barriers assessment is complete and what the final barrier is.

    Args:
        act_4_answer_1: User's barrier from Q1
        act_4_answer_2: User's barrier from Q2
        act_4_answer_3: User's barrier from Q3
        act_4_answer_4: User's barrier from Q4
        act_3_type: User's motivation type (for tiebreaker)

    Returns:
        Tuple of (should_finalize, final_barrier)
        - should_finalize: True if we have 4 answers
        - final_barrier: The determined barrier type
    """
    # Need all 4 answers to finalize
    if not act_4_answer_2 or act_4_answer_2 == "":
        return (False, "")

    # We have 4 answers, time to finalize
    # Call the logic function to determine final barrier
    final_barrier = compute_final_barrier(act_4_answer_1, act_4_answer_2, act_3_type)

    return (True, final_barrier)
def update_act_4_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update barriers metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored act_4_mapping
    2. Updates act_4_answer_1, act_4_answer_2, act_4_answer_3, or act_4_answer_4 based on current turn
    3. Updates last_act_4 to most recent answer
    4. Checks if barriers should be finalized
    5. Updates confirm_act_4 and act_4_type
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    act_4_block = stage_meta.get("act_4", {}) or {}
    act_4_state = dict(act_4_block.get("state", {}) or {})
    act_4_meta = dict(act_4_block.get("metadata", {}) or {})

    # Get stored mapping from previous agent run
    prev_act_4_mapping = act_4_state.get("act_4_mapping", [])
    prev_options = act_4_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = act_4_state.get("turn", 0)

    # Only process if we have a mapping and the user actually answered
    if not prev_act_4_mapping or not user_answer:
        return state

    # Classify the user's answer
    chosen_act_4 = get_user_chosen_act_4(user_answer, prev_options, prev_act_4_mapping)

    print(
        f"DEBUG - update_act_4_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_act_4='{chosen_act_4}'")

    # Update barrier based on which question they just answered
    # Update barrier based on which question they just answered
    # Update support preference based on which question they just answered
    if current_turn == 1:
        act_4_state["act_4_answer_1"] = chosen_act_4
        act_4_state["last_act_4"] = chosen_act_4
        act_4_state["last_theme"] = chosen_act_4
    elif current_turn == 2:
        act_4_state["act_4_answer_2"] = chosen_act_4
        act_4_state["last_act_4"] = chosen_act_4
        act_4_state["last_theme"] = chosen_act_4

    # Get both support preferences
    act_4_answer_1 = act_4_state.get("act_4_answer_1", "")
    act_4_answer_2 = act_4_state.get("act_4_answer_2", "")

    # Get act_3_type for finalization logic
    act_3_block = stage_meta.get("act_3", {}) or {}
    act_3_meta = act_3_block.get("metadata", {}) or {}
    act_3_type = act_3_meta.get("act_3_type", "")

    # Check if we should finalize (after 2 questions)
    should_finalize, final_barrier = should_finalize_act_4(
        act_4_answer_1, act_4_answer_2, "", "", act_3_type
    )

    print(f"DEBUG - should_finalize={should_finalize}, final_barrier='{final_barrier}'")

    if should_finalize:
        # We're done with barriers questions
        act_4_meta["confirm_act_4"] = "clear"
        act_4_meta["act_4_type"] = final_barrier
    else:
        # Keep asking questions
        act_4_meta["confirm_act_4"] = "unclear"
        act_4_meta["act_4_type"] = ""

    # Write back the updated metadata and state
    stage_meta["act_4"] = {
        "metadata": act_4_meta,
        "state": act_4_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }

def merge_stage_meta(left: Optional[Dict], right: Optional[Dict]) -> Dict:
    """Deep merge stage_meta dictionaries"""
    import copy

    result = copy.deepcopy(left) if left else {}
    if right:
        for key, value in right.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = {**result[key], **value}
            else:
                result[key] = value
    return result


class AgentState(TypedDict, total=False):
    """Global conversation state shared between supervisor and all agents."""

    # Conversation history + last user message
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str

    # Raw collected info per info_type (e.g. "connection", "wellness", "motivation")
    collected_info: Dict[str, List[Any]]

    # Routing + control flags
    next_agent: str
    agent_index: int
    exchanges_with_current: int
    last_agent: str

    stage_meta: Annotated[Dict[str, Dict[str, Any]], merge_stage_meta]

    # Last answer options shown to the user (for the UI to render)
    last_options: List[str]

    # Ad / acquisition context (where the user came from, campaign, theme, etc.)
    ad_data: AdData

    # Persistent, cross-agent user profile.
    user_profile: Dict[str, Any]
def compute_final_intent_from_state(
    state_block: Dict[str, Any],
    ad_theme: str,
    ) -> str:
    """
    Deterministic implementation of the TURN 4 rules for intent finalization.

    Inputs:
      - state_block: the 'state' dict from stage_meta['connection_intent']['state']
      - ad_theme: ad_data['ad_theme'] (may be "")

    Output:
      - act_1_type: one of "unsure", "self_expression", "wellness",
        "skill_growth", "ambition", "belonging", or "mixed".
    """
    theme_1 = state_block.get("theme_1", "") or ""
    theme_2 = state_block.get("theme_2", "") or ""
    theme_3 = state_block.get("theme_3", "") or ""
    ad_theme = ad_theme or ""

    # STEP 1 – how many times was the user explicitly "unsure"?
    themes = [theme_1, theme_2, theme_3]
    uncertain_total = sum(1 for t in themes if t == "unsure")

    # RULE 1 – UNSURE CASE (highest priority)
    if uncertain_total >= 2:
        return "unsure"

    # RULE 2 – REPEATING THEME CASE
    # look at [theme_1, theme_2, theme_3, ad_theme], ignoring "" and "unsure"
    seq = [theme_1, theme_2, theme_3, ad_theme]
    filtered = [t for t in seq if t and t != "unsure"]

    if filtered:
        counts = Counter(filtered)
        max_count = max(counts.values())
        if max_count >= 2:
            # candidates with the max count
            candidates = [t for t, c in counts.items() if c == max_count]

            # tie-break: prefer the most recent user-driven theme:
            # theme_3 > theme_2 > theme_1 > ad_theme
            priority_order = [theme_3, theme_2, theme_1, ad_theme]
            for p in priority_order:
                if p in candidates:
                    return p

            # fallback if something is weird
            return candidates[0]

    # RULE 3 – MIXED CASE (fallback)
    return "mixed"


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def render_tmpl(tmpl: str, **ctx) -> str:
    return (tmpl or "").format_map(_SafeDict(**ctx))


# ============================================================================
# BASE AGENT CLASS (Abstract)
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, system: str, human_template: str, info_type: str):
        self.name = name
        self.system = system
        self.human_template = human_template
        self.info_type = info_type
        self.use_openai = USE_OPENAI

    @abstractmethod
    def get_mock_response(self, user_text: str, state: AgentState) -> str:
        """Provide a mock response when OpenAI is unavailable."""
        raise NotImplementedError

    def build_context(self, state: AgentState, user_text: str) -> Dict[str, Any]:
        """Build context dictionary for template rendering."""
        stage_meta = state.get("stage_meta", {}) or {}
        my_block = stage_meta.get(self.name, {}) or {}
        my_meta = my_block.get("metadata", {}) or {}
        my_state = my_block.get("state", {}) or {}

        history = state.get("messages", [])
        hist_str = "\n".join([f"{m.__class__.__name__}: {m.content}" for m in history[-10:]])

        collected = state.get("collected_info", {}) or {}
        collected_str = json.dumps(collected, ensure_ascii=False, indent=2)

        ad_data = state.get("ad_data", {}) or {}
        ad_ctx_str = json.dumps(ad_data, ensure_ascii=False, indent=2)

        state_json_str = json.dumps(my_state, ensure_ascii=False, indent=2)

        # Determine selected_option
        selected_option = ""
        if history:
            last_msg = history[-1]
            if isinstance(last_msg, HumanMessage):
                selected_option = last_msg.content.strip()

        ctx = {
            "history": hist_str,
            "user_text": user_text,
            "collected_info": collected_str,
            "ad_context": ad_ctx_str,
            "state_json": state_json_str,
            "selected_option": selected_option,
            "ad_theme": ad_data.get("ad_theme", ""),
        }

        # FOR HOOK AGENT: provide ad context
        if self.info_type == "hook":
            ad_data = state.get("ad_data", {}) or {}
            ctx["ad_theme"] = ad_data.get("ad_theme", "")
            ctx["ad_description"] = ad_data.get("ad_description", "")
            return ctx

        # FOR CONNECTION_INTENT AGENT: compute last_act_1 and question_direction
        if self.info_type == "connection_intent":
            stage_meta = state.get("stage_meta", {}) or {}
            act_1_block = stage_meta.get("act_1", {}) or {}
            act_1_state = act_1_block.get("state", {}) or {}

            current_turn = act_1_state.get("turn", 0)
            ad_data = state.get("ad_data", {}) or {}
            ad_theme = ad_data.get("ad_theme", "")

            theme_1 = act_1_state.get("theme_1", "")
            theme_2 = act_1_state.get("theme_2", "")
            last_theme = act_1_state.get("last_theme", "")

            # Determine last_act_1
            if current_turn == 0:
                # First turn - use ad_theme
                last_act_1 = ad_theme
            else:
                # Use last_theme if available, otherwise ad_theme
                last_act_1 = last_theme if last_theme else ad_theme

            # Compute question_direction
            question_direction = compute_question_direction(theme_1, theme_2, current_turn + 1)

            # Determine focus_type based on turn number
            # Turn 1: aspiration broad
            # Turn 2: aspiration focused
            # Turn 3: identity shift broad
            # Turn 4: identity shift focused
            if current_turn + 1 == 1:
                focus_type = "aspiration"
            elif current_turn + 1 == 2:
                focus_type = "aspiration"
            elif current_turn + 1 == 3:
                focus_type = "identity"
            elif current_turn + 1 == 4:
                focus_type = "identity"
            else:
                focus_type = "aspiration"  # Fallback
                # ADD THESE DEBUG LINES
            print(f"🔍 DEBUG BUILD_CONTEXT:")
            print(f"   current_turn = {current_turn}")
            print(f"   next_turn (current_turn + 1) = {current_turn + 1}")
            print(f"   question_direction = {question_direction}")
            print(f"   focus_type = {focus_type}")
            print(f"   last_theme = {last_act_1}")

            ctx["last_theme"] = last_act_1
            ctx["question_mode"] = question_direction
            ctx["focus_type"] = focus_type

            # FOR EMOTIONAL_TONE AGENT: get last_act_1 from connection_intent results
        if self.info_type == "emotional_tone":
            stage_meta = state.get("stage_meta", {}) or {}

            # Get the finalized intent from connection_intent agent
            act_1_block = stage_meta.get("act_1", {}) or {}
            act_1_meta = act_1_block.get("metadata", {}) or {}
            act_1_state = act_1_block.get("state", {}) or {}

            # Get Act 2 state
            act_2_block = stage_meta.get("act_2", {}) or {}
            act_2_state = act_2_block.get("state", {}) or {}
            current_turn = act_2_state.get("turn", 0)

            # Use the finalized act_1_type, or fall back to last_theme, or ad_theme
            act_1_type = act_1_meta.get("act_1_type", "")
            last_theme = act_1_state.get("last_theme", "")
            ad_data = state.get("ad_data", {}) or {}
            ad_theme = ad_data.get("ad_theme", "")

            # Determine last_act_1 for emotional tone questions
            if act_1_type:
                last_act_1 = act_1_type  # Use finalized intent
            elif last_theme:
                last_act_1 = last_theme
            else:
                last_act_1 = ad_theme

            ctx["last_act_1"] = last_act_1

            # Get act1_direction from Act 1 Q2 (focused aspiration)
            theme_2 = act_1_state.get("theme_2", "")
            act1_direction = theme_2 if theme_2 else ad_theme
            ctx["act1_direction"] = act1_direction

            # Compute question_direction for Act 2
            question_direction = compute_question_direction_act_2(current_turn + 1)

            # Determine focus_type based on turn number
            # Turn 1: learning pattern broad
            # Turn 2: learning pattern focused
            # Turn 3: engagement pattern broad
            # Turn 4: engagement pattern focused
            if current_turn + 1 == 1:
                focus_type = "learning"
            elif current_turn + 1 == 2:
                focus_type = "learning"
            elif current_turn + 1 == 3:
                focus_type = "engagement"
            elif current_turn + 1 == 4:
                focus_type = "engagement"
            else:
                focus_type = "learning"  # Fallback

            ctx["question_mode"] = question_direction
            ctx["focus_type"] = focus_type



        # FOR MOTIVATION AGENT: compute last_act_3 and question_direction
        if self.info_type == "motivation":

            act2_state = stage_meta.get("act_2", {}).get("state", {})

            act2_emo_1 = act2_state.get("act_2_emo_1", "")
            act2_emo_2 = act2_state.get("act_2_emo_2", "")
            act2_emo_3 = act2_state.get("act_2_emo_3", "")
            act2_emo_4 = act2_state.get("act_2_emo_4", "")

            derived = self.compute_act2_fields(
                act2_emo_1,
                act2_emo_2,
                act2_emo_3,
                act2_emo_4
            )

            stage_meta.setdefault("act_3", {}).setdefault("state", {}).update(derived)
            state["stage_meta"] = stage_meta

            stage_meta = state.get("stage_meta", {}) or {}

            # Get the finalized intent from connection_intent agent
            act_1_block = stage_meta.get("act_1", {}) or {}
            act_1_meta = act_1_block.get("metadata", {}) or {}
            act_1_type = act_1_meta.get("act_1_type", "")

            # Get motivation state
            act_3_block = stage_meta.get("act_3", {}) or {}
            act_3_state = act_3_block.get("state", {}) or {}

            current_turn = act_3_state.get("turn", 0)

            act_3_answer_1 = act_3_state.get("act_3_answer_1", "")
            act_3_answer_2 = act_3_state.get("act_3_answer_2", "")
            act_3_answer_3 = act_3_state.get("act_3_answer_3", "")
            last_act_3_state = act_3_state.get("last_act_3", "")

            # Determine last_act_3
            if current_turn == 0:
                # First turn - use act_1_type from intent agent
                last_act_3 = act_1_type if act_1_type else ""
            else:
                # Use last_act_3 if available, otherwise act_1_type
                last_act_3 = last_act_3_state if last_act_3_state else act_1_type

            # Get act_1 state for additional context
            act_1_state = act_1_block.get("state", {}) or {}
            act1_direction = act_1_state.get("theme_1", "")  # emotional direction from Act 1
            act1_identity = act_1_state.get("theme_3", "")  # identity shift from Act 1

            # Determine question_mode and focus_type based on turn number
            # Turn 1: broad internal_fear
            # Turn 2: focused internal_fear
            # Turn 3: broad emotional_pattern
            # Turn 4: focused emotional_pattern
            if current_turn + 1 == 1:
                question_mode = "broad"
                focus_type = "internal_fear"
            elif current_turn + 1 == 2:
                question_mode = "focused"
                focus_type = "internal_fear"
            elif current_turn + 1 == 3:
                question_mode = "broad"
                focus_type = "emotional_pattern"
            elif current_turn + 1 == 4:
                question_mode = "focused"
                focus_type = "emotional_pattern"
            else:
                question_mode = "broad"
                focus_type = "internal_fear"  # Fallback

            ctx["last_theme"] = last_act_3
            ctx["question_mode"] = question_mode
            ctx["focus_type"] = focus_type
            ctx["act1_direction"] = act1_direction
            ctx["act1_identity"] = act1_identity
        if self.info_type == "barriers":
            stage_meta = state.get("stage_meta", {}) or {}

            # Get the finalized motivation from motivation agent
            act_3_block = stage_meta.get("act_3", {}) or {}
            act_3_meta = act_3_block.get("metadata", {}) or {}
            act_3_type = act_3_meta.get("act_3_type", "")

            # Get Act 1 identity for context
            act_1_block = stage_meta.get("act_1", {}) or {}
            act_1_state = act_1_block.get("state", {}) or {}
            act1_identity = act_1_state.get("theme_3", "")  # Identity from Q3

            # Get Act 2 derived psychographic fields
            act_2_block = stage_meta.get("act_2", {}) or {}
            act_2_state = act_2_block.get("state", {}) or {}

            act2_learning_style = act_2_state.get("act2_learning_style", "")
            act2_engagement_style = act_2_state.get("act2_engagement_style", "")
            act2_final_tone = act_2_state.get("act2_final_tone", "")

            # Get barriers state
            act_4_block = stage_meta.get("act_4", {}) or {}
            act_4_state = act_4_block.get("state", {}) or {}

            current_turn = act_4_state.get("turn", 0)

            act_4_answer_1 = act_4_state.get("act_4_answer_1", "")
            act_4_answer_2 = act_4_state.get("act_4_answer_2", "")
            act_4_answer_3 = act_4_state.get("act_4_answer_3", "")
            last_act_4_state = act_4_state.get("last_act_4", "")

            # Determine last_act_4
            if current_turn == 0:
                # First turn - use act_3_type from motivation agent
                last_act_4 = act_3_type if act_3_type else ""
            else:
                # Use last_act_4 if available, otherwise act_3_type
                last_act_4 = last_act_4_state if last_act_4_state else act_3_type

            # Compute question_direction (this becomes question_mode)
            question_mode = compute_question_direction_act_4(
                act_4_answer_1, current_turn + 1
            )

            # Determine focus_type based on turn number
            # Both turns focus on "support"
            focus_type = "support"

            # Add all fields to context
            ctx["last_theme"] = last_act_4
            ctx["question_mode"] = question_mode
            ctx["focus_type"] = focus_type
            ctx["act1_identity"] = act1_identity
            ctx["act2_learning_style"] = act2_learning_style
            ctx["act2_engagement_style"] = act2_engagement_style
            ctx["act2_final_tone"] = act2_final_tone

            # Keep old field names for backwards compatibility
            ctx["last_act_4"] = last_act_4
            ctx["question_direction"] = question_mode

        if self.info_type == "summary":
            print("DEBUG - Summary agent building context")
            stage_meta = state.get("stage_meta", {}) or {}
            print(f"DEBUG - stage_meta keys: {list(stage_meta.keys())}")
            # Get intent data
            act_1_block = stage_meta.get("act_1", {}) or {}
            act_1_meta = act_1_block.get("metadata", {}) or {}
            act_1_state = act_1_block.get("state", {}) or {}

            # Get tone data
            act_2_block = stage_meta.get("act_2", {}) or {}
            act_2_meta = act_2_block.get("metadata", {}) or {}
            act_2_state = act_2_block.get("state", {}) or {}

            # Get motivation data
            act_3_block = stage_meta.get("act_3", {}) or {}
            act_3_meta = act_3_block.get("metadata", {}) or {}
            act_3_state = act_3_block.get("state", {}) or {}

            # Get barriers data
            act_4_block = stage_meta.get("act_4", {}) or {}
            act_4_meta = act_4_block.get("metadata", {}) or {}
            act_4_state = act_4_block.get("state", {}) or {}

            # Add intent data to context
            ctx["act_1_type"] = act_1_meta.get("act_1_type", "")
            ctx["theme_1"] = act_1_state.get("theme_1", "")
            ctx["theme_2"] = act_1_state.get("theme_2", "")
            ctx["theme_3"] = act_1_state.get("theme_3", "")

            # Add tone data to context
            ctx["act_2_emo_tone"] = act_2_meta.get("act_2_emo_tone", "")
            ctx["act_2_emo_1"] = act_2_state.get("act_2_emo_1", "")
            ctx["act_2_emo_2"] = act_2_state.get("act_2_emo_2", "")

            # Add motivation data to context
            ctx["act_3_type"] = act_3_meta.get("act_3_type", "")
            ctx["act_3_answer_1"] = act_3_state.get("act_3_answer_1", "")
            ctx["act_3_answer_2"] = act_3_state.get("act_3_answer_2", "")
            ctx["act_3_answer_3"] = act_3_state.get("act_3_answer_3", "")
            ctx["act_3_answer_4"] = act_3_state.get("act_3_answer_4", "")
            # Add barriers data to context
            ctx["act_4_type"] = act_4_meta.get("act_4_type", "")
            ctx["act_4_answer_1"] = act_4_state.get("act_4_answer_1", "")
            ctx["act_4_answer_2"] = act_4_state.get("act_4_answer_2", "")

        return ctx


    def prepare_messages(self, ctx: Dict[str, str], state: AgentState) -> List[BaseMessage]:
        """Prepare the message list for LLM invocation."""
        sys_content = render_tmpl(self.system, **ctx)
        human_content = render_tmpl(self.human_template, **ctx)

        recent_msgs = list(state.get("messages", []))[-8:]
        return [SystemMessage(content=sys_content)] + recent_msgs + [HumanMessage(content=human_content)]

    def generate_response(self, user_text: str, state: AgentState) -> AgentResponse:
        """Generate a structured response using the LLM."""
        if not self.use_openai or ChatOpenAI is None:
            mock_text = self.get_mock_response(user_text, state)
            return AgentResponse(
                question_text=mock_text,
                options=[],
                option_mapping=["", "", "", ""],
                metadata={},
                state={},
            )

        ctx = self.build_context(state, user_text)
        msgs = self.prepare_messages(ctx, state)

        try:
            llm = ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=2500,
            )

            # Use strict schema for connection_intent agent
            # Use strict schema for hook agent
            if self.info_type == "hook":
                structured_llm = llm.with_structured_output(HookResponse)
                strict_response: HookResponse = structured_llm.invoke(msgs)
                return AgentResponse(
                    affirmation="",
                    question_text=strict_response.hook_text,
                    options=[],
                    option_mapping=["", "", "", ""],
                    metadata={"hook_text": strict_response.hook_text},
                    state={},
                )

            # Use strict schema for connection_intent agent

                # Use strict schema for connection_intent agent
            elif self.info_type == "connection_intent":
                structured_llm = llm.with_structured_output(ConnectionIntentResponse)

                # DEBUG: Log raw LLM response before validation
                try:
                    strict_response: ConnectionIntentResponse = structured_llm.invoke(msgs)
                    print(f"✅ DEBUG ACT_1 - Validation PASSED")
                    print(f"   Options count: {len(strict_response.options)}")
                    print(f"   Mapping count: {len(strict_response.option_mapping)}")
                    print(f"   Options: {strict_response.options}")
                    print(f"   Mapping: {strict_response.option_mapping}")
                except Exception as e:
                    print(f"❌ DEBUG ACT_1 - Validation FAILED")
                    print(f"   Error: {e}")
                    # Try to get the raw response without validation
                    try:
                        raw_llm = llm.invoke(msgs)
                        print(f"   Raw LLM response: {raw_llm}")
                    except:
                        pass
                    raise e  # Re-raise to trigger fallback

                # Convert to AgentResponse format
                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=strict_response.options,
                    option_mapping=strict_response.option_mapping,
                    metadata={},
                    state={}
                )
            # Use strict schema for emotional_tone agent



            # Use strict schema for emotional_tone agent
            elif self.info_type == "emotional_tone":
                structured_llm = llm.with_structured_output(EmotionalToneResponse)
                strict_response: EmotionalToneResponse = structured_llm.invoke(msgs)

                # Convert to AgentResponse format
                # Store response_format and scale-specific data in metadata
                act_2_metadata = {
                    "response_format": strict_response.response_format
                }

                # Add scale-specific fields if this is a scale question
                if strict_response.response_format == "scale":
                    act_2_metadata["scale_range"] = strict_response.scale_range
                    act_2_metadata["scale_labels"] = strict_response.scale_labels or {}
                    act_2_metadata["scale_mapping"] = strict_response.scale_mapping or {}

                # Pad act_2_emotional_mapping to always have 4 items (AgentResponse requirement)
                act_2_emotional_mapping = list(strict_response.act_2_emotional_mapping or [])
                while len(act_2_emotional_mapping) < 4:
                    act_2_emotional_mapping.append("")  # Pad with empty strings

                # Pad options to match if needed
                options = list(strict_response.options or [])
                while len(options) < 4:
                    options.append("")  # Pad with empty strings

                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=options,
                    option_mapping=act_2_emotional_mapping,
                    metadata=act_2_metadata,
                    state={}
                )
            # Use strict schema for motivation agent
            elif self.info_type == "motivation":
                structured_llm = llm.with_structured_output(MotivationResponse)
                strict_response: MotivationResponse = structured_llm.invoke(msgs)
                # Convert to AgentResponse format
                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=strict_response.options,
                    option_mapping=strict_response.act_3_mapping,
                    metadata={},
                    state={}
                )
            # Use strict schema for barriers agent
            elif self.info_type == "barriers":
                structured_llm = llm.with_structured_output(BarriersResponse)
                strict_response: BarriersResponse = structured_llm.invoke(msgs)
                # Convert to AgentResponse format
                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=strict_response.options,
                    option_mapping=strict_response.act_4_mapping,
                    metadata={},
                    state={}
                )
            # Use strict schema for summary agent
            elif self.info_type == "summary":
                structured_llm = llm.with_structured_output(SummaryResponse)
                strict_response: SummaryResponse = structured_llm.invoke(msgs)
                # Convert to AgentResponse format with summary data stored in metadata
                response = AgentResponse(
                    question_text=strict_response.summary_text,
                    options=[],
                    option_mapping=["", "", "", ""],  # Required field with 4 items
                    metadata={
                        "summary_text": strict_response.summary_text,
                        "recommendations": strict_response.recommendations
                    },
                    state={}
                )

            else:
                # Use flexible schema for other agents
                structured_llm = llm.with_structured_output(AgentResponse)
                response: AgentResponse = structured_llm.invoke(msgs)

            return response

        except Exception as e:
            print(f"Error with structured output: {e}")
            import traceback
            traceback.print_exc()
            # Return a proper fallback response for hook agent
            if self.info_type == "hook":
                return AgentResponse(
                    affirmation="",
                    question_text="Welcome! Let's discover what resonates with you.",
                    options=[],
                    option_mapping=["", "", "", ""],
                    metadata={"hook_text": "Welcome! Let's discover what resonates with you.", "error": str(e)},
                    state={},
                )

            # Return a proper fallback response for connection_intent agent

            if self.info_type == "connection_intent":
                return AgentResponse(
                    affirmation="I'd love to understand what brings you here today.",
                    question_text="When I think about what I'm hoping for right now, I feel most drawn to...",
                    options=[
                        "Expressing myself creatively",
                        "Learning something new",
                        "Finding some calm",
                        "I'm not quite sure yet"
                    ],
                    option_mapping=["self_expression", "skill_growth", "wellness", "unsure"],
                    metadata={"error": str(e)},
                    state={},
                )
            # Return a proper fallback response for emotional_tone agent

            elif self.info_type == "emotional_tone":
                return AgentResponse(
                    affirmation="It's completely okay to feel however you're feeling right now.",
                    question_text="When I think about taking this step, I feel...",
                    options=[
                        "Excited and ready",
                        "Curious but cautious",
                        "A bit overwhelmed",
                        "I'm not sure yet"
                    ],
                    option_mapping=["positive", "neutral", "tense", "unsure"],
                    metadata={"error": str(e)},
                    state={},
                )
            # Return a proper fallback response for motivation agent
            elif self.info_type == "motivation":
                return AgentResponse(
                    affirmation="Understanding what truly drives you is so important.",
                    question_text="When I think about why I want to do this, I feel most motivated by...",
                    options=[
                        "Seeing myself improve and grow",
                        "Expressing myself creatively",
                        "Finding calm and balance",
                        "I'm still figuring it out"
                    ],
                    option_mapping=["progress", "expression", "relief", "unsure"],
                    metadata={"error": str(e)},
                    state={},
                )
            # Return a proper fallback response for barriers agent
            elif self.info_type == "barriers":
                return AgentResponse(
                    affirmation="It's important to understand what might be in your way.",
                    question_text="When I think about what stops me from moving forward, I notice...",
                    options=[
                        "Struggling to stay consistent",
                        "Feeling overwhelmed by everything",
                        "Not having enough time or energy",
                        "I'm still figuring it out"
                    ],
                    option_mapping=["consistency", "overwhelm", "time_energy", "unsure"],
                    metadata={"error": str(e)},
                    state={},
                )
            # Return a proper fallback response for summary agent
            elif self.info_type == "summary":
                fallback_summary_text = "Thank you for sharing your journey with me. I've captured your insights and I'm here to support you."
                fallback_recommendations = [
                    "Take one small step today toward your goal",
                    "Reflect on what you've learned about yourself",
                    "Reach out for support when you need it"
                ]

                return AgentResponse(
                    question_text=fallback_summary_text,
                    options=[],
                    option_mapping=["", "", "", ""],
                    metadata={
                        "error": str(e),
                        "summary_text": fallback_summary_text,
                        "recommendations": fallback_recommendations
                    },
                    state={},
                )

            else:
                # Fallback for other agents
                mock_text = self.get_mock_response(user_text, state)
                return AgentResponse(
                    question_text=mock_text,
                    options=[],
                    option_mapping=["", "", "", ""],
                    metadata={"error": str(e)},
                    state={},
                )

    def process(self, state: AgentState) -> AgentState:
        """Main processing method called by LangGraph."""
        user_text = state.get("user_input", "")

        # Call the LLM (or mock) to get a structured AgentResponse
        response = self.generate_response(user_text, state)
        print(f"\n\n====== {self.name} AgentResponse ======\n", response, "\n===========================\n\n")

        # Build user-facing text: affirmation + question + enumerated options
        display_text = ""
        # Handle affirmation - skip act_1's first affirmation only
        # Handle affirmation - skip act_1's first affirmation only
        if response.affirmation:
            # Check if this is act_1's first question (turn 0, before increment)
            stage_meta = state.get("stage_meta", {}) or {}
            act_1_block = stage_meta.get("act_1", {}) or {}
            act_1_state = act_1_block.get("state", {}) or {}
            act_1_turn = act_1_state.get("turn", 0)

            # Skip affirmation only if it's act_1 and turn is 0 (first question, before increment)
            if not (self.info_type == "connection_intent" and act_1_turn == 0):
                display_text = format_affirmation(response.affirmation) + "\n\n"



        # Check if this is from hook agent and format accordingly
        if self.info_type == "hook" and response.question_text:
            display_text += format_hook_text(response.question_text)
        else:
            display_text += response.question_text or ""

        if response.options:
            lines = []
            for i, opt in enumerate(response.options):
                label = chr(65 + i)  # A, B, C, ...
                lines.append(f"{label}) {opt}")
            display_text = display_text.rstrip() + "\n\n" + "\n".join(lines)

        import copy

        # ------------------------------------------------------------------
        # STAGE META MERGE + HARD STATE MACHINE FOR connection_intent
        # ------------------------------------------------------------------
        # Start from previous stage_meta snapshot
        stage_meta_prev = copy.deepcopy(state.get("stage_meta", {}) or {})

        # Previous block for this agent
        old_block = stage_meta_prev.get(self.name) or {}
        old_meta = dict(old_block.get("metadata") or {})
        old_state = dict(old_block.get("state") or {})

        # New stuff coming from the model
        resp_meta = dict(response.metadata or {})
        resp_state = dict(response.state or {})

        # Start new_meta/new_state as copies of old, then merge in model output
        new_meta = dict(old_meta)
        new_state = dict(old_state)

        if resp_meta:
            new_meta.update(resp_meta)
        if resp_state:
            new_state.update(resp_state)
        # ---------- HOOK LOGIC ----------
        if self.info_type == "hook":
            # Hook agent just displays the message and sets status to clear
            hook_text = resp_meta.get("hook_text", "")
            new_meta["hook_status"] = "clear"
            new_meta["hook_text"] = hook_text
            new_state["hook_displayed"] = True

            stage_meta_prev[self.name] = {
                "metadata": new_meta,
                "state": new_state,
            }


        # ---------- HARD TURN / LEVEL / METADATA LOGIC FOR CONNECTION INTENT ----------
        # ---------- SIMPLIFIED CONNECTION_INTENT LOGIC ----------
        if self.info_type == "connection_intent":
            # Get current turn from old_state
            old_turn = old_state.get("turn", 0)
            try:
                old_turn = int(old_turn)
            except (TypeError, ValueError):
                old_turn = 0

            # Increment turn
            current_turn = old_turn + 1
            new_state["turn"] = current_turn

            # Initialize fields on turn 1
            if current_turn == 1:
                ad_data = state.get("ad_data", {}) or {}
                ad_theme = ad_data.get("ad_theme", "")
                new_state["ad_theme"] = ad_theme
                new_state["theme_1"] = ""
                new_state["theme_2"] = ""
                new_state["theme_3"] = ""
                new_state["theme_4"] = ""
                new_state["last_theme"] = ad_theme  # Start with ad_theme

            # Set level based on turn
            if current_turn == 1:
                new_state["last_level"] = "L1"
            elif current_turn == 2:
                new_state["last_level"] = "L2"
            elif current_turn == 3:
                new_state["last_level"] = "L3"
            elif current_turn == 4:
                new_state["last_level"] = "L4"
            else:
                new_state["last_level"] = "L4"  # Fallback

            # Store option_mapping and options for next turn's classification
            if response.option_mapping:
                new_state["option_mapping"] = response.option_mapping
            if response.options:
                new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_act_1_metadata_after_answer)
            new_meta.setdefault("confirm_act_1", "unclear")
            new_meta.setdefault("act_1_type", "")
            new_state.setdefault("last_theme", "")
            stage_meta_prev[self.name] = {
                "metadata": new_meta,
                "state": new_state,
            }

        # ---------- HARD TURN / LEVEL / METADATA LOGIC FOR EMOTIONAL_TONE ----------
        if self.info_type == "emotional_tone":
            # Get current turn from old_state
            old_turn = old_state.get("turn", 0)
            try:
                old_turn = int(old_turn)
            except (TypeError, ValueError):
                old_turn = 0

            # Increment turn
            current_turn = old_turn + 1
            new_state["turn"] = current_turn

            # Initialize fields on turn 1
            if current_turn == 1:
                new_state["act_2_emo_1"] = ""
                new_state["act_2_emo_2"] = ""
                new_state["act_2_emo_3"] = ""
                new_state["act_2_emo_4"] = ""
                new_state["response_format_1"] = ""
                new_state["response_format_2"] = ""
                new_state["response_format_3"] = ""
                new_state["response_format_4"] = ""
                new_state["last_theme"] = ""

            # Get response format from metadata
            response_format = resp_meta.get("response_format", "multiple_choice")

            # Store response format for this turn
            if current_turn == 1:
                new_state["response_format_1"] = response_format
            elif current_turn == 2:
                new_state["response_format_2"] = response_format
            elif current_turn == 3:
                new_state["response_format_3"] = response_format
            elif current_turn == 4:
                new_state["response_format_4"] = response_format

            # Handle different response formats
            if response_format == "scale":
                # For scale: store scale_range and scale_mapping
                new_state["scale_range"] = resp_meta.get("scale_range", "")
                new_state["scale_labels"] = resp_meta.get("scale_labels", {})
                new_state["scale_mapping"] = resp_meta.get("scale_mapping", {})
                # Don't store options/act_2_emotional_mapping for scales
            else:
                # For multiple_choice and yes_no: store options and mapping
                if response.option_mapping:
                    new_state["act_2_emotional_mapping"] = response.option_mapping
                if response.options:
                    new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_emotional_act_2_metadata_after_answer)
            new_meta.setdefault("confirm_act_2", "unclear")
            new_meta.setdefault("act_2_emo_tone", "unclear")

            stage_meta_prev[self.name] = {
                "metadata": new_meta,
                "state": new_state,
            }
            # ---------- HARD TURN / LEVEL / METADATA LOGIC FOR MOTIVATION ----------
        if self.info_type == "motivation":
            # Get current turn from old_state
            old_turn = old_state.get("turn", 0)
            try:
                old_turn = int(old_turn)
            except (TypeError, ValueError):
                old_turn = 0

            # Increment turn
            current_turn = old_turn + 1
            new_state["turn"] = current_turn

            # Initialize fields on turn 1
            if current_turn == 1:
                ad_data = state.get("ad_data", {}) or {}
                ad_theme = ad_data.get("ad_theme", "")
                new_state["ad_theme"] = ad_theme
                new_state["act_3_answer_1"] = ""
                new_state["act_3_answer_2"] = ""
                new_state["act_3_answer_3"] = ""
                new_state["act_3_answer_4"] = ""
                new_state["last_act_3"] = ""  # Will be set after first answer

            # Set level based on turn
            if current_turn == 1:
                new_state["last_level"] = "L1"
            elif current_turn == 2:
                new_state["last_level"] = "L2"
            elif current_turn == 3:
                new_state["last_level"] = "L3"
            elif current_turn == 4:
                new_state["last_level"] = "L4"
            else:
                new_state["last_level"] = "L5"

            # Store act_3_mapping and options from LLM response
            if response.option_mapping:
                new_state["act_3_mapping"] = response.option_mapping
            if response.options:
                new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_act_3_metadata_after_answer)
            new_meta.setdefault("confirm_act_3", "unclear")
            new_meta.setdefault("act_3_type", "")
            new_state.setdefault("last_act_3", "")

            stage_meta_prev[self.name] = {
                "metadata": new_meta,
                "state": new_state,
            }
            # ---------- HARD TURN / LEVEL / METADATA LOGIC FOR BARRIERS ----------
        if self.info_type == "barriers":
            # Get current turn from old_state
            old_turn = old_state.get("turn", 0)
            try:
                old_turn = int(old_turn)
            except (TypeError, ValueError):
                old_turn = 0
            # Increment turn
            current_turn = old_turn + 1
            new_state["turn"] = current_turn

            # Initialize fields on turn 1
            if current_turn == 1:
                ad_data = state.get("ad_data", {}) or {}
                ad_theme = ad_data.get("ad_theme", "")
                new_state["ad_theme"] = ad_theme
                new_state["act_4_answer_1"] = ""
                new_state["act_4_answer_2"] = ""
                new_state["last_act_4"] = ""  # Will be set after first answer

            # Set level based on turn
            if current_turn == 1:
                new_state["last_level"] = "L1"
            elif current_turn == 2:
                new_state["last_level"] = "L2"
            else:
                new_state["last_level"] = "L2"  # Only 2 turns

            # Store act_4_mapping and options from LLM response
            if response.option_mapping:
                new_state["act_4_mapping"] = response.option_mapping
            if response.options:
                new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_act_4_metadata_after_answer)
            new_meta.setdefault("confirm_act_4", "unclear")
            new_meta.setdefault("act_4_type", "")
            new_state.setdefault("last_act_4", "")

            stage_meta_prev[self.name] = {
                "metadata": new_meta,
                "state": new_state,
            }

            # ---------- SUMMARY AGENT LOGIC ----------
        # ---------- SUMMARY AGENT LOGIC ----------
        if self.info_type == "summary":
            # Summary runs only once (turn = 1)
            new_state["turn"] = 1

            # Immediately mark as complete
            new_meta["confirm_summary"] = "clear"
            # Store summary_text and recommendations from response
            # Check if it's in metadata (fallback case) or as attributes (success case)
            # Always read from metadata (where the data actually is)
            new_meta["summary_text"] = response.metadata.get("summary_text", "")
            new_meta["recommendations"] = response.metadata.get("recommendations", [])


            stage_meta_prev[self.name] = {
                "metadata": new_meta,
                "state": new_state,
            }
        # ----- SPECIAL CASE: LOGIC FINALIZER -----
        if self.name == "logic_finalize":
            # Get the most up-to-date state for connection_intent
            ci_state_block = dict(new_state)

            # Pull ad_theme from ad_data in the global state
            ad_data = state.get("ad_data", {}) or {}
            ad_theme = ad_data.get("ad_theme", "") or ""

            # Compute final act_1_type deterministically
            final_act_1_type = compute_final_intent_from_state(
                ci_state_block,
                ad_theme=ad_theme,
            )

            # Finalizer always sets confirm_act_1 = "clear"
            new_meta["act_1_type"] = final_act_1_type
            new_meta["confirm_act_1"] = "clear"

            # Mark this as the final (TURN 4) snapshot
            new_state["turn"] = 4
            new_state["last_level"] = "L4"

            # Update last_theme if we have a clear non-unsure, non-mixed category
            if final_act_1_type not in ("unsure", "mixed", ""):
                new_state["last_theme"] = final_act_1_type
        if self.info_type == "connection_tone":
            # 1) Hard-control the tone turn counter in Python (ignore LLM's turn)
            #    We treat turns as PHASES: 1 = Q1, 2 = Q2, 3 = FINAL, 4 = DONE.
            old_turn = old_state.get("turn", 1)
            try:
                incoming_turn = int(old_turn)
            except (TypeError, ValueError):
                incoming_turn = 1

            if incoming_turn <= 0:
                incoming_turn = 1

            # Map incoming_turn -> next_turn
            # YAML says: turn 1 = Q1, turn 2 = Q2, turn 3 = FINAL, turn 4 = DONE
            if incoming_turn <= 1:
                next_turn = 2  # Q1 outputs turn=2
            elif incoming_turn == 2:
                next_turn = 3  # Q2 outputs turn=3
            elif incoming_turn == 3:
                next_turn = 4  # FINAL outputs turn=4
            else:
                next_turn = 4  # Stay at 4 for safety

            print(f"DEBUG connection_tone - incoming_turn={incoming_turn}, next_turn will be={next_turn}")

            new_state["turn"] = next_turn

            # Simple level mapping for tone (just for debug / UI)
            if next_turn == 2:
                new_state["last_level"] = "L1"
            elif next_turn == 3:
                new_state["last_level"] = "L2"
            else:
                new_state["last_level"] = "L3"

            # 2) Enforce per-turn emotional tone invariants
            #
            # TURN 1→2 (Q1 phase):
            #   - act_2_emo_1 MUST be "" (no classification yet)
            #   - act_2_emo_2 MUST be ""
            #
            # TURN 2→3 (Q2 phase):
            #   - act_2_emo_1 may be filled by the LLM from Q1
            #   - act_2_emo_2 MUST stay ""
            #
            # TURN 3→4 (FINAL phase):
            #   - LLM sets act_2_emo_2 + final act_2_emo_tone / confirm_act_2
            #
            # TURN 4+ (DONE phase):
            #   - Everything stays as-is

            if incoming_turn <= 1:
                # Q1 phase - first tone question → wipe any premature tone labels
                new_state["act_2_emo_1"] = ""
                new_state["act_2_emo_2"] = ""
                new_state.setdefault("act_2_emo_3", "")
                # Ensure metadata fields exist
                new_meta.setdefault("act_2_emo_tone", "unclear")
                new_meta.setdefault("emo_act_2_type", "")
                new_meta.setdefault("confirm_act_2", "unclear")

            elif incoming_turn == 2:
                # Q2 phase - act_2_emo_1 should already be set by LLM from Q1
                # Keep act_2_emo_2 blank until Turn 3
                new_state.setdefault("act_2_emo_1", "")  # keep it if already set
                new_state["act_2_emo_2"] = ""  # not set yet
                new_state.setdefault("act_2_emo_3", "")
                # Keep tone unclear during Q2
                new_meta.setdefault("act_2_emo_tone", "unclear")
                new_meta.setdefault("emo_act_2_type", "")
                new_meta.setdefault("confirm_act_2", "unclear")

            elif incoming_turn == 3:
                # FINAL phase - LLM should set act_2_emo_2 + final classification
                new_state.setdefault("act_2_emo_1", "")
                new_state.setdefault("act_2_emo_2", "")  # LLM MUST populate this now
                new_state.setdefault("act_2_emo_3", "")

                # Safety check: if LLM didn't set act_2_emo_2, log warning
                if not new_state.get("act_2_emo_2"):
                    print("WARNING: act_2_emo_2 not set by LLM during FINAL phase")

                # Safety check: if LLM didn't set confirm_act_2 to "clear", force it
                if new_meta.get("confirm_act_2") != "clear":
                    print("WARNING: LLM didn't set confirm_act_2='clear' in FINAL phase, forcing it")
                    new_meta["confirm_act_2"] = "clear"

                    # If LLM also didn't set final act_2_emo_tone, compute it as fallback
                    if not new_meta.get("act_2_emo_tone") or new_meta.get("act_2_emo_tone") == "unclear":
                        emo_1 = new_state.get("act_2_emo_1", "")
                        emo_2 = new_state.get("act_2_emo_2", "")

                        if emo_1 and emo_2:
                            if emo_1 == emo_2 and emo_1 != "unsure":
                                final_tone = emo_1
                            elif emo_1 == "unsure" and emo_2 == "unsure":
                                final_tone = "unsure"
                            else:
                                final_tone = "mixed"
                        elif emo_1:
                            final_tone = emo_1
                        elif emo_2:
                            final_tone = emo_2
                        else:
                            final_tone = "unsure"

                        print(f"FALLBACK: Computing final tone as '{final_tone}' from emo_1={emo_1}, emo_2={emo_2}")
                        new_meta["act_2_emo_tone"] = final_tone
                        new_meta["emo_act_2_type"] = final_tone

            else:
                # DONE phase (turn >= 4) - don't ask more questions
                new_state.setdefault("act_2_emo_1", "")
                new_state.setdefault("act_2_emo_2", "")
                new_state.setdefault("act_2_emo_3", "")
                # Ensure confirm_act_2 stays "clear"
                if new_meta.get("confirm_act_2") != "clear":
                    print("WARNING: confirm_act_2 not 'clear' in DONE phase, fixing")
                    new_meta["confirm_act_2"] = "clear"

            # 3) Copy canonical intent from connection_intent into tone metadata
            act_1_block = stage_meta_prev.get("connection_intent") or {}
            act_1_meta = act_1_block.get("metadata") or {}

            canonical_act_1_type = act_1_meta.get("act_1_type")
            canonical_confirm_act_1 = act_1_meta.get("confirm_act_1")

            if canonical_act_1_type is not None:
                new_meta["act_1_type"] = canonical_act_1_type
            if canonical_confirm_act_1 is not None:
                new_meta["confirm_act_1"] = canonical_confirm_act_1

            # 4) Normalize behavioral_signal to a safe default if it's weird/empty
            allowed_signals = {
                "first_timer",
                "hobbyist",
                "busy_adult",
                "ambitious_learner",
                "mixed",
            }
            if new_meta.get("behavioral_signal") not in allowed_signals:
                new_meta["behavioral_signal"] = "mixed"

            # 5) Write back the corrected tone block
            stage_meta_prev["connection_tone"] = {
                "metadata": new_meta,
                "state": new_state,
            }

        print(f"DEBUG - {self.name} - metadata: {new_meta}")
        print(f"DEBUG - {self.name} - state: {new_state}")

        # Update collected info
        merged_collected = dict(state.get("collected_info", {}))

        if user_text:
            merged_collected.setdefault(self.info_type, []).append(user_text)

        return AgentState(
            messages=[AIMessage(content=display_text)],
            collected_info=merged_collected,
            stage_meta=stage_meta_prev,
            exchanges_with_current=state.get("exchanges_with_current", 0) + 1,
            last_agent=self.name,
        )

    def compute_act2_fields(self, act_2_emo_1: str, act_2_emo_2: str,
                            act_2_emo_3: str, act_2_emo_4: str) -> Dict[str, str]:
        """
        Convert Act 2 emotional mappings into Act 3–ready human-readable signals.
        """

        unified_map = {
            # momentum-driven
            "fast_starter": "momentum_driven",
            "early_wins": "momentum_driven",
            "micro_wins": "momentum_driven",
            "confidence_boost": "momentum_driven",

            # steady
            "slow_steady": "steady",
            "reflective": "steady",
            "reflective_only": "steady",
            "structure_seeker": "steady",
            "meaning_first": "steady",
            "guided_support": "steady",
            "flexible_pacing": "steady",
            "clarity_seeker": "steady",
            "consistency_seeker": "steady",
            "passive_learner": "steady",
            "short_session": "steady",
            "unclear_steps": "steady",
            "too_many_choices": "steady",

            # exploratory
            "explorer": "exploratory",
            "hands_on": "exploratory",
            "visually_rewarding": "exploratory",
            "playful_learning": "exploratory",
            "curated_exploration": "exploratory",
            "variety_seeker": "exploratory",

            # friction-sensitive
            "overwhelm": "friction_sensitive",
            "time_pressure": "friction_sensitive",
            "setup_friction": "friction_sensitive",
            "lose_momentum": "friction_sensitive",
            "low_initial_progress": "friction_sensitive",
            "distraction_prone": "friction_sensitive",

            # ambiguous
            "unsure": "ambiguous"
        }

        HUMAN_MEANINGS = {
            "steady": "You approach learning in a calm, thoughtful, and consistent way.",
            "exploratory": "You learn best by exploring, experimenting, and following curiosity.",
            "momentum_driven": "You stay motivated by forward movement — early progress and quick wins help you stay engaged.",
            "friction_sensitive": "You thrive when barriers are removed and the path feels smooth and supported.",
            "ambiguous": "Your learning preferences are still emerging and haven’t settled into a clear pattern yet.",
            "mixed": "Your learning and engagement styles pull in different directions, creating a mixed emotional rhythm."
        }

        # Determine learning category
        raw_learning = act_2_emo_1 or "unsure"
        learning_category = unified_map.get(raw_learning, "ambiguous")

        # Determine engagement category
        raw_engagement = act_2_emo_3 or "unsure"
        engagement_category = unified_map.get(raw_engagement, "ambiguous")

        # Determine final tone
        if learning_category == "ambiguous" or engagement_category == "ambiguous":
            final_category = "ambiguous"
        elif learning_category == engagement_category:
            final_category = learning_category
        else:
            final_category = "mixed"

        # Convert categories → human sentences
        return {
            "act2_learning_style": HUMAN_MEANINGS[learning_category],
            "act2_engagement_style": HUMAN_MEANINGS[engagement_category],
            "act2_final_tone": HUMAN_MEANINGS[final_category]
        }


class SupervisorDecision(PydanticV1BaseModel):
    next_agent: str = PydanticV1Field(
        description=(
            "Which agent should run next. One of: "
            "connection_intent, connection_tone, wellness, creativity, monetization, FINISH."
        )
    )
    reason: str = PydanticV1Field(
        description="Short explanation of why you chose this route.",
        default=""
    )

    class Config:
        extra = "forbid"
# ============================================================================
# CONCRETE AGENT IMPLEMENTATIONS
# ============================================================================

class ConnectionAgent(BaseAgent):
    """Stage 1: Connection & Curiosity."""

    def get_mock_response(self, user_text: str, state: AgentState) -> str:
        return (
            "Hey there! I'm Sparky – quick check-in before we explore together.\n"
            "What feels most true right now?\n"
            "A) I want to restart my creative spark\n"
            "B) I'm curious but hesitant\n"
            "C) I'm overwhelmed and need calm\n"
            "D) Not sure yet\n\n"
        )


class WellnessAgent(BaseAgent):
    """Stage 2: Wellness Deep Dive."""

    def get_mock_response(self, user_text: str, state: AgentState) -> str:
        return (
            "I hear you're interested in wellness and mindfulness. Let's explore what that means for you.\n"
            "What outcome would matter most right now?\n"
            "A) Reducing daily stress and anxiety\n"
            "B) Improving sleep quality\n"
            "C) Building emotional resilience\n"
            "D) Just exploring different approaches\n"
        )


class CreativityAgent(BaseAgent):
    """Stage 2: Creativity Deep Dive."""

    def get_mock_response(self, user_text: str, state: AgentState) -> str:
        return (
            "Love that you're here to explore your creative side!\n"
            "What type of creative expression calls to you?\n"
            "A) Visual arts (drawing, painting, design)\n"
            "B) Writing and storytelling\n"
            "C) Music or sound creation\n"
            "D) I'm open to discovering what resonates\n"
        )


class MonetizationAgent(BaseAgent):
    """Stage 2: Side Hustle / Monetization."""

    def get_mock_response(self, user_text: str, state: AgentState) -> str:
        return (
            "Awesome! Let's talk about turning skills into income.\n"
            "What's your primary goal?\n"
            "A) Build a side income stream ($500-2000/month)\n"
            "B) Transition to a new career path\n"
            "C) Freelance with skills I already have\n"
            "D) Just exploring what's possible\n"
        )


# ============================================================================
# AGENT FACTORY
# ============================================================================

def create_agent(agent_type: str, name: str, system: str, human_template: str, info_type: str) -> BaseAgent:
    """Factory function to create the appropriate agent subclass based on type."""
    agent_map = {
        # Stage 1 – Connection split
        "connection_intent": ConnectionAgent,   # new intent agent
        "connection_tone": ConnectionAgent,     # new emotional tone agent

        # Stage 2 – Deep dives
        "wellness": WellnessAgent,
        "wellness_deep": WellnessAgent,  # alias

        "creativity": CreativityAgent,
        "creativity_deep": CreativityAgent,  # alias

        "monetization": MonetizationAgent,
        "side_hustle": MonetizationAgent,  # alias
    }

    agent_class = agent_map.get(agent_type, ConnectionAgent)
    return agent_class(name, system, human_template, info_type)


# ============================================================================
# SUPERVISOR CLASS
# ============================================================================

class Supervisor:
    def __init__(self, agents: Dict[str, BaseAgent], prompts: Dict[str, str]):
        self.agents = agents
        self.agent_keys = list(agents.keys())
        self.system = prompts.get("system", "")
        self.human_template = prompts.get("human_template", "")
        # LLM is only used if we have OpenAI + both supervisor prompts
        self.use_openai = USE_OPENAI and bool(self.system and self.human_template)

    # ------------------------------------------------------------------
    # ORIGINAL RULE-BASED ROUTING (FALLBACK + SAFETY NET)
    # ------------------------------------------------------------------
    def route(self, state: AgentState) -> AgentState:
        """
        Deterministic, rule-based routing.
        This is kept as a safe fallback when LLM is unavailable or fails.
        """
        collected = state.get("collected_info", {})
        stage_meta = state.get("stage_meta", {}) or {}

        # --- Read INTENT block (act_1) ---
        act_1_block = stage_meta.get("act_1", {}) or {}
        act_1_meta = act_1_block.get("metadata", {}) or {}

        # --- Read TONE block (connection_tone) ---
        act_2_block = stage_meta.get("connection_tone", {}) or {}
        act_2_meta = act_2_block.get("metadata", {}) or {}

        # Status fields
        act_1_status = (
                act_1_meta.get("confirm_act_1")
                or act_1_meta.get("act_1_status")
                or ""
        ).lower()
        act_1_type = (act_1_meta.get("act_1_type") or "").lower()
        act_2_emo_tone = (
                act_2_meta.get("act_2_emo_tone")
                or act_2_meta.get("emotional_tone")
                or "unclear"
        ).lower()

        # Turn counts per agent (fallback to counts of collected answers)
        act_1_turns = len(collected.get("act_1", []))
        act_2_turns = len(collected.get("connection_tone", []))
        total_turns = act_1_turns + act_2_turns

        ad_data = state.get("ad_data", {}) or {}
        ad_theme = ad_data.get("ad_creative_theme", "").lower()

        print(
            f"DEBUG SUPERVISOR (fallback) - act_1_turns: {act_1_turns}, "
            f"act_2_turns: {act_2_turns}, total_turns: {total_turns}, "
            f"act_1_status: {act_1_status}, act_2_emo_tone: {act_2_emo_tone}, "
            f"ad_theme: {ad_theme}, act_1_type: {act_1_type}"
        )

        # Stage 1 exit readiness (roughly mirror YAML logic)
        ready = (act_1_status in ("clear", "unsure")) and (act_2_emo_tone not in ("resistant", "unclear"))

        # --- If ready for deep stage or we've hit max turns, route out of Stage 1 ---
        if ready or total_turns >= 6:
            routing_map = {
                "wellness": ["wellness_deep", "wellness"],
                "mindfulness": ["wellness_deep", "wellness"],
                "therapeutic": ["wellness_deep", "wellness"],
                "creativity": ["creativity_deep", "creativity"],
                "self-expression": ["creativity_deep", "creativity"],
                "side_hustle": ["monetization", "side_hustle"],
                "achievement": ["monetization", "side_hustle"],
                "ambition": ["monetization", "side_hustle"],
            }

            # Prefer explicit act_1_type, fall back to ad theme
            route_key = act_1_type if act_1_type in routing_map else ad_theme

            if route_key in routing_map:
                for possible_agent in routing_map[route_key]:
                    if possible_agent in self.agent_keys:
                        print(f"DEBUG SUPERVISOR (fallback) - Routing to {possible_agent} agent")
                        return {
                            "agent_index": self.agent_keys.index(possible_agent),
                            "next_agent": possible_agent,
                            "exchanges_with_current": 0,
                        }

            print(
                f"DEBUG SUPERVISOR (fallback) - FINISHING Stage 1 "
                f"(ready={ready}, total_turns={total_turns})"
            )
            return {"next_agent": "FINISH", "exchanges_with_current": 0}

        # --- Otherwise, we're still in Stage 1: pick the right connection sub-agent ---

        # 1) If intent is still unclear → keep working with intent agent
        if act_1_status in ("", "unclear"):
            if "act_1" in self.agent_keys:
                print("DEBUG SUPERVISOR (fallback) - Continuing with act_1 agent")
                return {
                    "agent_index": self.agent_keys.index("act_1"),
                    "next_agent": "act_1",
                    "exchanges_with_current": 0,
                }

        # 2) If intent is set but tone is unclear → move to tone agent
        if act_2_emo_tone in ("", "unclear"):
            if "connection_tone" in self.agent_keys:
                print("DEBUG SUPERVISOR (fallback) - Routing to connection_tone agent")
                return {
                    "agent_index": self.agent_keys.index("connection_tone"),
                    "next_agent": "connection_tone",
                    "exchanges_with_current": 0,
                }

        # 3) If neither sub-agent is available for some reason → finish
        print("DEBUG SUPERVISOR (fallback) - No connection_intent/tone agent present → FINISH")
        return {"next_agent": "FINISH", "exchanges_with_current": 0}

    # ------------------------------------------------------------------
    # NEW: LLM-BASED SUPERVISOR CONTEXT BUILDING
    # ------------------------------------------------------------------
    def build_context(self, state: AgentState) -> Dict[str, str]:
        """
        Build the context passed into the supervisor prompts.
        Extracts key variables from stage_meta for easy access by the LLM.
        """
        stage_meta = state.get("stage_meta", {}) or {}
        # Extract hook status
        hook_block = stage_meta.get("hook", {}) or {}
        hook_meta = hook_block.get("metadata", {}) or {}
        hook_status = hook_meta.get("hook_status", "unclear")

        # Extract intent info
        act_1_block = stage_meta.get("act_1", {}) or {}
        act_1_meta = act_1_block.get("metadata", {}) or {}
        act_1_state = act_1_block.get("state", {}) or {}

        act_1_status = act_1_meta.get("confirm_act_1", "unclear")
        act_1_turn = act_1_state.get("turn", 0)

        # Extract tone info - try emotional_tone first, fallback to connection_tone
        act_2_block = stage_meta.get("act_2", {}) or stage_meta.get("connection_tone", {}) or {}
        act_2_meta = act_2_block.get("metadata", {}) or {}
        act_2_state = act_2_block.get("state", {}) or {}

        confirm_act_2 = act_2_meta.get("confirm_act_2", "unclear")
        act_2_turn = act_2_state.get("turn", 0)
        act_2_emo_1 = act_2_state.get("act_2_emo_1", "")
        act_2_emo_2 = act_2_state.get("act_2_emo_2", "")
        act_2_emo_tone = act_2_meta.get("act_2_emo_tone", "unclear")

        # Extract motivation info
        act_3_block = stage_meta.get("act_3", {}) or {}
        act_3_meta = act_3_block.get("metadata", {}) or {}
        act_3_state = act_3_block.get("state", {}) or {}

        confirm_act_3 = act_3_meta.get("confirm_act_3", "unclear")
        act_3_turn = act_3_state.get("turn", 0)
        act_3_answer_1 = act_3_state.get("act_3_answer_1", "")
        act_3_answer_2 = act_3_state.get("act_3_answer_2", "")
        act_3_answer_3 = act_3_state.get("act_3_answer_3", "")
        act_3_answer_4 = act_3_state.get("act_3_answer_4", "")
        act_3_type = act_3_meta.get("act_3_type", "unclear")

        # Extract barriers info
        act_4_block = stage_meta.get("act_4", {}) or {}
        act_4_meta = act_4_block.get("metadata", {}) or {}
        act_4_state = act_4_block.get("state", {}) or {}

        confirm_act_4 = act_4_meta.get("confirm_act_4", "unclear")
        act_4_turn = act_4_state.get("turn", 0)
        act_4_answer_1 = act_4_state.get("act_4_answer_1", "")
        act_4_answer_2 = act_4_state.get("act_4_answer_2", "")
        act_4_type = act_4_meta.get("act_4_type", "unclear")
        last_act_4_state = act_4_state.get("last_act_4", "")

        # Extract summary info
        summary_block = stage_meta.get("summary", {}) or {}
        summary_meta = summary_block.get("metadata", {}) or {}

        confirm_summary = summary_meta.get("confirm_summary", "unclear")
        summary_turn = summary_meta.get("turn", 0)

        return {
            # Raw JSON (for reference if supervisor needs full context)
            "stage_meta": json.dumps(stage_meta, ensure_ascii=False),
            "collected_info": json.dumps(state.get("collected_info", {}), ensure_ascii=False),
            "ad_data": json.dumps(state.get("ad_data", {}), ensure_ascii=False),
            "user_profile": json.dumps(state.get("user_profile", {}), ensure_ascii=False),

            # Extracted variables for easy access by supervisor LLM
            "last_agent": state.get("last_agent", "") or "",
            "hook_status": hook_status,
            # Intent variables
            "act_1_status": act_1_status,
            "act_1_turn": str(act_1_turn),

            # Tone variables
            "confirm_act_2": confirm_act_2,
            "act_2_turn": str(act_2_turn),
            "act_2_emo_1": act_2_emo_1,
            "act_2_emo_2": act_2_emo_2,
            "act_2_emo_tone": act_2_emo_tone,

            # Motivation variables
            "confirm_act_3": confirm_act_3,
            "act_3_turn": str(act_3_turn),
            "act_3_answer_1": act_3_answer_1,
            "act_3_answer_2": act_3_answer_2,
            "act_3_answer_3": act_3_answer_3,
            "act_3_answer_4": act_3_answer_4,
            "act_3_type": act_3_type,

            # Barriers variables
            "confirm_act_4": confirm_act_4,
            "act_4_turn": str(act_4_turn),
            "act_4_answer_1": act_4_answer_1,
            "act_4_answer_2": act_4_answer_2,
            "act_4_type": act_4_type,



            # Summary variables
            "confirm_summary": confirm_summary,
            "summary_turn": str(summary_turn),

        }

    def prepare_messages(self, ctx: Dict[str, str]) -> List[BaseMessage]:
            """
            Prepare the System + Human messages for the supervisor LLM,
            using the YAML system + human_template.
            """
            sys_msg = SystemMessage(content=render_tmpl(self.system, **ctx))
            human_msg = HumanMessage(content=render_tmpl(self.human_template, **ctx))
            return [sys_msg, human_msg]

        # ------------------------------------------------------------------
        # NEW: LLM-DRIVEN SUPERVISOR NODE

    def process(self, state: AgentState) -> AgentState:
        """
        LLM-based routing entrypoint used by LangGraph.

        - Reads the full AgentState
        - Calls the supervisor LLM (if available) with SupervisorDecision schema
        - Writes next_agent / agent_index back into state
        - Falls back to rule-based route() on error or when LLM is disabled
        """
        # No LLM available or no supervisor prompts → fallback to rule-based routing
        if not self.use_openai or ChatOpenAI is None:
            return self.route(state)

        ctx = self.build_context(state)
        msgs = self.prepare_messages(ctx)

        try:
            llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, max_tokens=512)
            decision: SupervisorDecision = (
                llm.with_structured_output(SupervisorDecision).invoke(msgs)
            )
        except Exception as e:
            print(f"Supervisor LLM error, falling back to route(): {e}")
            import traceback
            traceback.print_exc()
            return self.route(state)

        next_agent = (decision.next_agent or "").lower()
        print(
            f"DEBUG SUPERVISOR LLM - next_agent={next_agent}, "
            f"reason={decision.reason}"
        )

        # Handle finish/end
        if next_agent in ("finish", "end"):
            return {"next_agent": "FINISH", "exchanges_with_current": 0}

        # If LLM suggested an unknown agent, fallback to deterministic routing
        if next_agent not in self.agent_keys:
            print(f"DEBUG SUPERVISOR LLM - unknown agent '{next_agent}', falling back to route()")
            return self.route(state)

        # Valid agent → return routing info
        return {
            "next_agent": next_agent,
            "agent_index": self.agent_keys.index(next_agent),
            "exchanges_with_current": 0,
        }

# ============================================================================
# GRAPH CREATION
# ============================================================================

def create_graph(agent_configs: Dict[str, Dict[str, Any]]):
    """Create the LangGraph workflow with agent instances."""
    agents: Dict[str, BaseAgent] = {}
    for key, cfg in agent_configs.items():
        agents[key] = create_agent(
            agent_type=key,
            name=cfg["name"],
            system=cfg["system"],
            human_template=cfg["human_template"],
            info_type=cfg["info_type"],
        )

    supervisor = Supervisor(agents, prompts=SUPERVISOR_PROMPTS)

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor.process)

    for k, a in agents.items():
        workflow.add_node(k, a.process)

    main_keys = list(agents.keys())

    def route_after_agent(state: AgentState) -> str:
        """
        After an agent runs, ALWAYS end this graph execution to wait for user input.
        The next graph.invoke() will start fresh from the supervisor based on updated metadata.
        """
        last_agent = state.get("last_agent", "")

        print(f"DEBUG - route_after_agent: last_agent={last_agent}, ending to wait for user")

        # ALWAYS END after agent runs - this allows UI to wait for user input
        return "END"

    def route_from_supervisor(state: AgentState) -> str:
        """
        Decide which node to go to next based on supervisor output,
        with HARD GUARDRAILS that override bad LLM decisions.
        """
        # What the LLM supervisor suggested
        nxt = (state.get("next_agent") or "").lower()

        # Read tone status from stage_meta (if present)
        stage_meta = state.get("stage_meta", {}) or {}
        act_2_block = stage_meta.get("connection_tone", {}) or {}
        act_2_meta = act_2_block.get("metadata", {}) or {}

        # New tone fields you want to use
        act_2_status = (act_2_meta.get("confirm_act_2") or "").lower()
        # act_2_status: "" or "unclear" means tone not finalized

        # 🔒 HARD RULE:
        # If supervisor tries to FINISH but tone is not confirmed yet,
        # FORCE it to go to connection_tone instead (if that agent exists).
        if nxt in ("finish", "end") and act_2_status in ("", "unclear"):
            if "connection_tone" in main_keys:
                print("DEBUG ROUTER - Overriding FINISH → connection_tone because confirm_act_2 is not clear.")
                return "connection_tone"

        # Normal FINISH handling (only when act_2_status is clear or we don't care)
        if nxt in ("finish", "end"):
            return "END"

        # Otherwise, route to the requested agent if it exists
        return nxt if nxt in main_keys else (main_keys[0] if main_keys else "END")

    # Map supervisor → next_agent (or END)
    edges_map = {k: k for k in main_keys}
    edges_map.update({"END": END})
    workflow.add_conditional_edges("supervisor", route_from_supervisor, edges_map)

    # Map each agent to its next node
    # Map each agent to its next node
    for k in main_keys:
        if k == "logic_finalize":
            # After logic_finalize, go back to supervisor within the same invoke
            workflow.add_edge("logic_finalize", "supervisor")
        else:
            # All agents (including connection_intent) end after running
            # This allows UI to wait for user input before next graph.invoke()
            workflow.add_edge(k, END)

    workflow.set_entry_point("supervisor")
    return workflow.compile()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_options(text: str) -> Tuple[str, List[str]]:
    """Extract A) B) C) D) options from text."""
    lines = text.splitlines()
    opt_pattern = re.compile(r"^\s*([A-Z])\)\s*(.+)")
    options: List[str] = []
    kept_lines: List[str] = []

    for line in lines:
        m = opt_pattern.match(line)
        if m:
            options.append(m.group(2).strip())
        else:
            kept_lines.append(line)

    clean_text = "\n".join(kept_lines).strip()
    return clean_text, options


def process_user_message(graph, state: AgentState, msg: str) -> Tuple[str, AgentState]:
    print(f"DEBUG - Input state stage_meta: {state.get('stage_meta', {})}")

    before = len(state.get("messages", []))
    state = {
        **state,
        "messages": list(state.get("messages", [])) + [HumanMessage(content=msg)],
        "user_input": msg,
    }

    #  Ensure tone block and emo fields exist BEFORE graph.invoke
    state = ensure_act_2_block(state)
    # Update intent metadata if user answered an intent question
    last_agent = state.get("last_agent", "")
    print(f"🔍 DEBUG - Full state keys: {list(state.keys())}")  # ADD THIS
    print(f"🔍 DEBUG - process_user_message: last_agent='{last_agent}', msg='{msg}'")  # ADD THIS

    # Update intent metadata if user answered an intent question
    if last_agent == "act_1" and msg:
        state = update_act_1_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After intent metadata update: confirm_act_1={state.get('stage_meta', {}).get('act_1', {}).get('metadata', {}).get('confirm_act_1')}")

    # Update emotional tone metadata if user answered an emotional tone question
    if last_agent == "act_2" and msg:
        state = update_act_2_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After emotional tone metadata update: confirm_act_2={state.get('stage_meta', {}).get('act_2', {}).get('metadata', {}).get('confirm_act_2')}")
        # Update motivation metadata if user answered a motivation question
    if last_agent == "act_3" and msg:
        state = update_act_3_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After motivation metadata update: confirm_act_3={state.get('stage_meta', {}).get('act_3', {}).get('metadata', {}).get('confirm_act_3')}")
        # Update barriers metadata if user answered a barriers question
    if last_agent == "act_4" and msg:
        state = update_act_4_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After barriers metadata update: confirm_act_4={state.get('stage_meta', {}).get('act_4', {}).get('metadata', {}).get('confirm_act_4')}")
    new_state = graph.invoke(state)
    # Extract the AI's response text from the new messages
    new_messages = new_state.get("messages", [])
    if len(new_messages) > len(state.get("messages", [])):
        # Get the last message (which should be the AI's response)
        last_msg = new_messages[-1]
        if isinstance(last_msg, AIMessage):
            full_text = last_msg.content
            # Extract options and clean text
            clean_text, options = extract_options(full_text)
            new_state["last_options"] = options
        else:
            clean_text = ""
    else:
        clean_text = ""
    return clean_text, new_state


# ============================================================================
# STREAMLIT UI
# ============================================================================

def _init_session():
    if "graph" not in st.session_state:
        st.session_state.graph = create_graph(AGENT_CONFIGS)

    if "ad_data" not in st.session_state:
        st.session_state.ad_data = get_ad_data_from_external_source()

    if "state" not in st.session_state:
        collected_info_init = {cfg["info_type"] for cfg in AGENT_CONFIGS.values()}

        base_state = AgentState(
            messages=[],
            user_input="",
            collected_info={k: [] for k in collected_info_init},
            next_agent="",
            agent_index=0,
            exchanges_with_current=0,
            last_agent="",
            stage_meta={
                "hook": {
                    "metadata": {"hook_status": "unclear", "hook_text": ""},
                    "state": {"hook_displayed": False}
                }
            },
            last_options=[],
            ad_data=st.session_state.ad_data,
            user_profile={},
        )

        # ✅ make sure connection_tone + emo_act_2_type + confirm_act_2 exist from the start
        base_state = ensure_act_2_block(base_state)

        st.session_state.state = base_state

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    st.set_page_config(page_title="Sparky – Enhanced Multi-Stage", layout="wide")
    st.title("🤖 Sparky – Enhanced Multi-Stage AI Salesman")
    _init_session()

    st.info("OpenAI active" if USE_OPENAI else "Mock mode", icon="✅" if USE_OPENAI else "⚠️")

    # Auto-start
    if not st.session_state.chat_history:
        first_state = st.session_state.state
        first_state["user_input"] = ""
        ai_text, new_state = process_user_message(st.session_state.graph, first_state, "")
        st.session_state.state = new_state
        st.session_state.chat_history.append({"role": "assistant", "content": ai_text})

    with st.sidebar:
        # Display ad data
        st.header("📢 Ad Context")
        ad_data = st.session_state.ad_data or {}
        st.markdown(f"**Ad Name:** {ad_data.get('ad_name', 'N/A')}")
        st.markdown(f"**Description:** {ad_data.get('ad_description', 'N/A')}")
        st.markdown(f"**Theme:** {ad_data.get('ad_theme', 'N/A')}")
        st.markdown("---")

        st.markdown("---")
        st.header("Conversation Insights")
        meta = st.session_state.state.get("stage_meta", {}) or {}

        # Hook block
        hook_block = meta.get("hook", {}) or {}
        hook_meta = hook_block.get("metadata", {}) or {}
        hook_status = hook_meta.get("hook_status", "unclear")
        hook_text = hook_meta.get("hook_text", "—")

        st.markdown("**🎣 Hook:**")
        st.markdown(f"- **Status:** {hook_status}")
        if hook_text != "—":
            st.markdown(f"- **Message:** {hook_text[:100]}...")  # Show first 100 chars
        st.markdown("---")

        # Intent block (act_1)
        act_1_block = meta.get("act_1", {}) or {}
        act_1_meta = act_1_block.get("metadata", {}) or {}
        act_1_state = act_1_block.get("state", {}) or {}

        # Tone block (act_2) - CHECK BOTH act_2 AND connection_tone
        act_2_block = meta.get("act_2", {}) or meta.get("connection_tone", {}) or {}
        act_2_meta = act_2_block.get("metadata", {}) or {}
        act_2_state = act_2_block.get("state", {}) or {}

        # Motivation block (act_3)
        act_3_block = meta.get("act_3", {}) or {}
        act_3_meta = act_3_block.get("metadata", {}) or {}
        act_3_state = act_3_block.get("state", {}) or {}

        # Barriers block (act_4)
        act_4_block = meta.get("act_4", {}) or {}
        act_4_meta = act_4_block.get("metadata", {}) or {}
        act_4_state = act_4_block.get("state", {}) or {}

        # Intent fields
        act_1_status = act_1_meta.get("confirm_act_1") or act_1_meta.get("act_1_status", "—")
        act_1_type = act_1_meta.get("act_1_type", "—")

        # Tone fields
        confirm_act_2 = act_2_meta.get("confirm_act_2") or "unclear"
        act_2_type = act_2_meta.get("act_2_emo_tone") or act_2_meta.get("emo_act_2_type") or "—"
        act_2_emo_1 = act_2_state.get("act_2_emo_1") or "—"
        act_2_emo_2 = act_2_state.get("act_2_emo_2") or "—"
        act_2_emo_3 = act_2_state.get("act_2_emo_3") or "—"

        # Motivation fields
        confirm_act_3 = act_3_meta.get("confirm_act_3") or "unclear"
        act_3_type = act_3_meta.get("act_3_type", "—")
        act_3_answer_1 = act_3_state.get("act_3_answer_1") or "—"
        act_3_answer_2 = act_3_state.get("act_3_answer_2") or "—"
        act_3_answer_3 = act_3_state.get("act_3_answer_3") or "—"
        act_3_answer_4 = act_3_state.get("act_3_answer_4") or "—"

        # Barriers fields
        confirm_act_4 = act_4_meta.get("confirm_act_4") or "unclear"
        act_4_type = act_4_meta.get("act_4_type", "—")
        act_4_answer_1 = act_4_state.get("act_4_answer_1") or "—"
        act_4_answer_2 = act_4_state.get("act_4_answer_2") or "—"
        act_4_answer_3 = act_4_state.get("act_4_answer_3") or "—"
        act_4_answer_4 = act_4_state.get("act_4_answer_4") or "—"

        # Display insights
        st.markdown(f"- **Confirm Act 1:** {act_1_status or '—'}")
        st.markdown(f"- **Act 1 Type:** {act_1_type or '—'}")

        # Act 1 Details - NEW SECTION
        st.markdown("---")
        st.markdown("**🎯 Act 1 Details:**")

        # Define variables first (get them from act_1_state)
        current_turn = act_1_state.get("turn", 0)
        last_theme = act_1_state.get("last_theme", "—")

        # Compute question_mode for CURRENT question being displayed
        if current_turn == 0:
            question_mode = "—"
            focus_type = "—"
        elif current_turn == 1:
            question_mode = "broad"
            focus_type = "aspiration"
        elif current_turn == 2:
            question_mode = "focused"
            focus_type = "aspiration"
        elif current_turn == 3:
            question_mode = "broad"
            focus_type = "identity"
        elif current_turn == 4:
            question_mode = "focused"
            focus_type = "identity"
        else:
            question_mode = "—"
            focus_type = "—"

        # Now display them
        st.markdown(f"- **Current Turn:** {current_turn}")
        st.markdown(f"- **Last Theme:** {last_theme}")
        st.markdown(f"- **Question Mode (current question):** {question_mode}")
        st.markdown(f"- **Focus Type (current question):** {focus_type}")

        # Show individual answers (as mappings) if available
        theme_1 = act_1_state.get("theme_1", "—")
        theme_2 = act_1_state.get("theme_2", "—")
        theme_3 = act_1_state.get("theme_3", "—")
        theme_4 = act_1_state.get("theme_4", "—")

        if any([theme_1 != "—", theme_2 != "—", theme_3 != "—", theme_4 != "—"]):
            st.markdown("**Answers:**")
            if theme_1 != "—":
                st.markdown(f"  - Q1: {theme_1}")
            if theme_2 != "—":
                st.markdown(f"  - Q2: {theme_2}")
            if theme_3 != "—":
                st.markdown(f"  - Q3: {theme_3}")
            if theme_4 != "—":
                st.markdown(f"  - Q4: {theme_4}")

        st.markdown("---")

        # Act 2 section
        st.markdown("**😊 Act 2 Details:**")

        # Get Act 2 turn info
        act_2_turn = act_2_state.get("turn", 0)
        act_2_last_theme = act_2_state.get("last_theme", "—")

        # Compute question_mode for CURRENT question being displayed
        if act_2_turn == 0:
            act_2_question_mode = "—"
            act_2_focus_type = "—"
        elif act_2_turn == 1:
            act_2_question_mode = "broad"
            act_2_focus_type = "learning"
        elif act_2_turn == 2:
            act_2_question_mode = "focused"
            act_2_focus_type = "learning"
        elif act_2_turn == 3:
            act_2_question_mode = "broad"
            act_2_focus_type = "engagement"
        elif act_2_turn == 4:
            act_2_question_mode = "focused"
            act_2_focus_type = "engagement"
        else:
            act_2_question_mode = "—"
            act_2_focus_type = "—"

        st.markdown(f"- **Current Turn:** {act_2_turn}")
        st.markdown(f"- **Last Theme:** {act_2_last_theme}")
        st.markdown(f"- **Question Mode (current question):** {act_2_question_mode}")
        st.markdown(f"- **Focus Type (current question):** {act_2_focus_type}")
        st.markdown(f"- **Confirm Act 2:** {confirm_act_2}")
        st.markdown(f"- **Act 2 Type:** {act_2_type}")

        # Show individual answers (as mappings) if available
        act_2_emo_1 = act_2_state.get("act_2_emo_1", "—")
        act_2_emo_2 = act_2_state.get("act_2_emo_2", "—")
        act_2_emo_3 = act_2_state.get("act_2_emo_3", "—")
        act_2_emo_4 = act_2_state.get("act_2_emo_4", "—")

        if any([act_2_emo_1 != "—", act_2_emo_2 != "—", act_2_emo_3 != "—", act_2_emo_4 != "—"]):
            st.markdown("**Answers:**")
            if act_2_emo_1 != "—":
                st.markdown(f"  - Q1: {act_2_emo_1}")
            if act_2_emo_2 != "—":
                st.markdown(f"  - Q2: {act_2_emo_2}")
            if act_2_emo_3 != "—":
                st.markdown(f"  - Q3: {act_2_emo_3}")
            if act_2_emo_4 != "—":
                st.markdown(f"  - Q4: {act_2_emo_4}")

        st.markdown("---")

        # Act 3 section
        st.markdown("**💪 Act 3:**")

        # Get Act 3 turn info
        act_3_turn = act_3_state.get("turn", 0)
        act_3_last_theme = act_3_state.get("last_theme", "—")

        # Compute question_mode for CURRENT question being displayed
        if act_3_turn == 0:
            act_3_question_mode = "—"
            act_3_focus_type = "—"
        elif act_3_turn == 1:
            act_3_question_mode = "broad"
            act_3_focus_type = "internal_fear"
        elif act_3_turn == 2:
            act_3_question_mode = "focused"
            act_3_focus_type = "internal_fear"
        elif act_3_turn == 3:
            act_3_question_mode = "broad"
            act_3_focus_type = "emotional_pattern"
        elif act_3_turn == 4:
            act_3_question_mode = "focused"
            act_3_focus_type = "emotional_pattern"
        else:
            act_3_question_mode = "—"
            act_3_focus_type = "—"

        st.markdown(f"- **Current Turn:** {act_3_turn}")
        st.markdown(f"- **Last Theme:** {act_3_last_theme}")
        st.markdown(f"- **Question Mode:** {act_3_question_mode}")
        st.markdown(f"- **Focus Type:** {act_3_focus_type}")
        st.markdown(f"- **Confirm Act 3:** {confirm_act_3}")
        st.markdown(f"- **Act 3 Type:** {act_3_type}")

        # Show individual answers if available
        if any([act_3_answer_1 != "—", act_3_answer_2 != "—", act_3_answer_3 != "—", act_3_answer_4 != "—"]):
            st.markdown("**Answers:**")
            if act_3_answer_1 != "—":
                st.markdown(f"  - Q1: {act_3_answer_1}")
            if act_3_answer_2 != "—":
                st.markdown(f"  - Q2: {act_3_answer_2}")
            if act_3_answer_3 != "—":
                st.markdown(f"  - Q3: {act_3_answer_3}")
            if act_3_answer_4 != "—":
                st.markdown(f"  - Q4: {act_3_answer_4}")

        st.markdown("---")

        # Act 4 section
        st.markdown("**🚧 Act 4:**")

        # Get Act 4 turn info
        act_4_turn = act_4_state.get("turn", 0)
        act_4_last_theme = act_4_state.get("last_theme", "—")

        # Compute question_mode for CURRENT question being displayed
        if act_4_turn == 0:
            act_4_question_mode = "—"
            act_4_focus_type = "—"
        elif act_4_turn == 1:
            act_4_question_mode = "broad"
            act_4_focus_type = "support"
        elif act_4_turn == 2:
            act_4_question_mode = "focused"
            act_4_focus_type = "support"
        else:
            act_4_question_mode = "—"
            act_4_focus_type = "—"

        st.markdown(f"- **Current Turn:** {act_4_turn}")
        st.markdown(f"- **Last Theme:** {act_4_last_theme}")
        st.markdown(f"- **Question Mode:** {act_4_question_mode}")
        st.markdown(f"- **Focus Type:** {act_4_focus_type}")
        st.markdown(f"- **Confirm Act 4:** {confirm_act_4}")
        st.markdown(f"- **Act 4 Type:** {act_4_type}")

        # Show individual answers if available
        if any([act_4_answer_1 != "—", act_4_answer_2 != "—"]):
            st.markdown("**Answers:**")
            if act_4_answer_1 != "—":
                st.markdown(f"  - Q1: {act_4_answer_1}")
            if act_4_answer_2 != "—":
                st.markdown(f"  - Q2: {act_4_answer_2}")

        st.markdown("---")
        # Summary block
        summary_block = meta.get("summary", {}) or {}
        summary_meta = summary_block.get("metadata", {}) or {}

        # Summary fields
        confirm_summary = summary_meta.get("confirm_summary") or "unclear"

        st.markdown(f"- **Confirm Summary:** {confirm_summary}")

        if st.button("Reset conversation"):
            # Clear state and chat history, then rerun app
            collected_info_init = {cfg["info_type"] for cfg in AGENT_CONFIGS.values()}
            st.session_state.state = AgentState(
                messages=[],
                user_input="",
                collected_info={k: [] for k in collected_info_init},
                next_agent="",
                agent_index=0,
                exchanges_with_current=0,
                last_agent="",
                stage_meta={},
                last_options=[],
                ad_data=st.session_state.ad_data,
                user_profile={},
            )
            st.session_state.chat_history = []
            st.rerun()

    # Show chat history
    for t in st.session_state.chat_history:
        with st.chat_message(t["role"]):
            st.markdown(t["content"], unsafe_allow_html=True)

    # ========== CHECK IF CONVERSATION IS COMPLETE ==========
    stage_meta = st.session_state.state.get("stage_meta", {}) or {}

    # Check hook status
    hook_block = stage_meta.get("hook", {}) or {}
    hook_meta = hook_block.get("metadata", {}) or {}
    hook_status = hook_meta.get("hook_status", "unclear")

    # Check intent status
    act_1_block = stage_meta.get("act_1", {}) or {}
    act_1_meta = act_1_block.get("metadata", {}) or {}
    confirm_act_1 = act_1_meta.get("confirm_act_1", "unclear")
    act_1_type = act_1_meta.get("act_1_type", "")

    # Check emotional tone status - try emotional_tone first, fallback to connection_tone
    act_2_block = stage_meta.get("act_2", {}) or stage_meta.get("connection_tone", {}) or {}
    act_2_meta = act_2_block.get("metadata", {}) or {}
    confirm_act_2 = act_2_meta.get("confirm_act_2", "unclear")
    act_2_emo_tone = act_2_meta.get("act_2_emo_tone", "")

    # Check motivation status
    act_3_block = stage_meta.get("act_3", {}) or {}
    act_3_meta = act_3_block.get("metadata", {}) or {}
    confirm_act_3 = act_3_meta.get("confirm_act_3", "unclear")
    act_3_type = act_3_meta.get("act_3_type", "")
    act_1_block = stage_meta.get("act_1", {}) or {}
    act_1_state = act_1_block.get("state", {}) or {}
    act1_identity = act_1_state.get("theme_3", "")  # Identity from Q3
    # Get Act 2 derived psychographic fields
    act_2_block = stage_meta.get("act_2", {}) or {}
    act_2_state = act_2_block.get("state", {}) or {}

    act2_learning_style = act_2_state.get("act2_learning_style", "")
    act2_engagement_style = act_2_state.get("act2_engagement_style", "")
    act2_final_tone = act_2_state.get("act2_final_tone", "")
    # Check barriers status
    act_4_block = stage_meta.get("act_4", {}) or {}
    act_4_meta = act_4_block.get("metadata", {}) or {}
    confirm_act_4 = act_4_meta.get("confirm_act_4", "unclear")
    act_4_type = act_4_meta.get("act_4_type", "")

    # Conversation is complete when all four are clear
    # Check summary status
    summary_block = stage_meta.get("summary", {}) or {}
    summary_meta = summary_block.get("metadata", {}) or {}
    confirm_summary = summary_meta.get("confirm_summary", "unclear")
    summary_text = summary_meta.get("summary_text", "")
    recommendations = summary_meta.get("recommendations", [])

    # Conversation is complete when all FIVE are clear (including summary)
    conversation_ended = (confirm_act_1 == "clear" and confirm_act_2 == "clear" and
                          confirm_act_3 == "clear" and confirm_act_4 == "clear" and
                          confirm_summary == "clear")

    # ========== SHOW BUTTONS OR COMPLETION MESSAGE ==========
    if conversation_ended:
        # Show completion message
        st.success(f"✅ **Conversation Complete!**")
        st.info(
            f"**Intent:** {act_1_type} | **Emotional Tone:** {act_2_emo_tone} | **Motivation:** {act_3_type} | **Barrier:** {act_4_type}")

        # Show summary if available
        if summary_text:
            st.markdown("### 📋 Your Personalized Summary")
            st.markdown(summary_text)

        # Show recommendations if available
        if recommendations:
            st.markdown("### 💡 Next Steps")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

        st.markdown("---")
        st.markdown("We've captured your preferences and will tailor your experience accordingly. 🎉")
    else:
        # Check if this is the hook message (no options)
        stage_meta = st.session_state.state.get("stage_meta", {}) or {}
        hook_block = stage_meta.get("hook", {}) or {}
        hook_meta = hook_block.get("metadata", {}) or {}
        hook_status = hook_meta.get("hook_status", "unclear")

        # If hook just displayed and status is clear, automatically trigger next agent
        if hook_status == "clear" and st.session_state.chat_history and st.session_state.chat_history[-1][
            "role"] == "assistant":
            last_msg_content = st.session_state.chat_history[-1]["content"]
            # Check if this is the hook message (contains the hook text)
            hook_text = hook_meta.get("hook_text", "")
            if hook_text and hook_text in last_msg_content:
                # Auto-trigger next turn (Act 1)
                ai_text, new_state = process_user_message(
                    st.session_state.graph, st.session_state.state, ""
                )
                st.session_state.state = new_state
                if ai_text:
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_text})
                st.rerun()

        # Buttons/Input for current options (only if conversation is ongoing)
        options = st.session_state.state.get("last_options", []) or []
        if options and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
            with st.chat_message("assistant"):
                # Determine response format from stage_meta
                stage_meta = st.session_state.state.get("stage_meta", {}) or {}

                # Determine which agent is currently active
                last_agent = st.session_state.state.get("last_agent", "")

                # Only check for flexible formats if emotional_tone agent is active
                if last_agent == "act_2":
                    # Check emotional_tone agent state for response format
                    act_2_block = stage_meta.get("act_2", {}) or {}
                    act_2_state = act_2_block.get("state", {}) or {}
                    act_2_turn = act_2_state.get("turn", 0)

                    # Get the response format for the current question
                    if act_2_turn == 1:
                        response_format = act_2_state.get("response_format_1", "multiple_choice")
                    elif act_2_turn == 2:
                        response_format = act_2_state.get("response_format_2", "multiple_choice")
                    else:
                        response_format = "multiple_choice"

                    # Get scale info if needed
                    scale_range = act_2_state.get("scale_range", "")
                    scale_labels = act_2_state.get("scale_labels", {})
                else:
                    # All other agents use standard multiple_choice format
                    response_format = "multiple_choice"
                    scale_range = ""
                    scale_labels = {}

                # Get scale info if needed
                scale_range = act_2_state.get("scale_range", "")
                scale_labels = act_2_state.get("scale_labels", {})

                user_response = None

                # RENDER BASED ON RESPONSE FORMAT
                if response_format == "scale":
                    # SCALE INPUT: Show slider
                    st.write("Rate your response:")

                    # Parse scale range
                    if scale_range and "-" in scale_range:
                        min_val, max_val = map(int, scale_range.split("-"))
                    else:
                        min_val, max_val = 1, 10  # Default

                    # Show labels if available
                    if scale_labels:
                        min_label = scale_labels.get("min", "")
                        max_label = scale_labels.get("max", "")
                        if min_label and max_label:
                            st.caption(f"**{min_val}:** {min_label} | **{max_val}:** {max_label}")

                    # Slider input
                    scale_value = st.slider(
                        "Select a value:",
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        key="scale_slider"
                    )

                    if st.button("Submit", key="scale_submit"):
                        user_response = str(scale_value)

                elif response_format == "yes_no":
                    # YES/NO INPUT: Show 2 buttons
                    st.write("Choose your answer:")
                    col1, col2 = st.columns(2)
                    if len(options) >= 2:
                        if col1.button(options[0], key="yes_no_0"):
                            user_response = options[0]
                        if col2.button(options[1], key="yes_no_1"):
                            user_response = options[1]

                else:
                    # MULTIPLE CHOICE: Show 4 buttons (default behavior)
                    st.write("Choose an option:")
                    cols = st.columns(len(options))
                    for i, opt in enumerate(options):
                        if cols[i].button(opt, key=f"opt_{i}"):
                            user_response = opt

                # PROCESS USER RESPONSE
                if user_response:
                    st.session_state.chat_history.append({"role": "user", "content": user_response})
                    ai_text, new_state = process_user_message(
                        st.session_state.graph, st.session_state.state, user_response
                    )
                    st.session_state.state = new_state

                    # Check if conversation just ended
                    new_stage_meta = new_state.get("stage_meta", {}) or {}
                    new_act_1_meta = new_stage_meta.get("act_1", {}).get("metadata", {})
                    new_act_2_meta = new_stage_meta.get("act_2", {}).get("metadata", {}) or new_stage_meta.get(
                        "connection_tone", {}).get("metadata", {})
                    new_act_3_meta = new_stage_meta.get("act_3", {}).get("metadata", {})
                    new_act_4_meta = new_stage_meta.get("act_4", {}).get("metadata", {})

                    new_confirm_act_1 = new_act_1_meta.get("confirm_act_1", "unclear")
                    new_confirm_act_2 = new_act_2_meta.get("confirm_act_2", "unclear")
                    new_confirm_act_3 = new_act_3_meta.get("confirm_act_3", "unclear")
                    new_confirm_act_4 = new_act_4_meta.get("confirm_act_4", "unclear")

                    # Check if conversation is complete (all 4 agents done)
                    if (new_confirm_act_1 == "clear" and new_confirm_act_2 == "clear" and
                            new_confirm_act_3 == "clear" and new_confirm_act_4 == "clear"):
                        # Don't add ai_text to history - we'll show completion message instead
                        pass
                    elif ai_text:
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_text})

                    st.rerun()

        # Free-text input (only if conversation is ongoing)
        prompt = st.chat_input("Type your message...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            ai_text, new_state = process_user_message(st.session_state.graph, st.session_state.state, prompt)
            st.session_state.state = new_state
            st.session_state.chat_history.append({"role": "assistant", "content": ai_text})
            with st.chat_message("assistant"):
                st.markdown(ai_text, unsafe_allow_html=True)
            st.rerun()


if __name__ == "__main__":
    main()



