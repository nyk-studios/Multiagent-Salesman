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

DEFAULT_CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", "new_config.yaml")
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
        "ad_theme": "skill_growth",
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
        "ad_theme": "wellness",
    },
    "cookies_asmr": {
        "ad_name": "Sensory Soother",
        "ad_description": (
            "This ad taps into sensory satisfaction and calming ASMR visuals, appealing to viewers seeking "
            "emotional relief, comfort, and a peaceful creative ritual. It attracts overwhelmed adults and "
            "sensory-sensitive people who want low-pressure, feel-good activities that help them unwind. "
            "The promise isn’t mastery — it’s gentle creativity and emotional soothing, framed as self-care."
        ),
        "ad_theme": "wellness",
    },
    "purpose_tiktok": {
        "ad_name": "Creative Purpose",
        "ad_description": (
            "A reflective, emotionally resonant ad that speaks to people feeling directionless, uninspired, "
            "or disconnected from themselves. It offers creativity as a path to meaning, identity renewal, "
            "and inner clarity. It attracts seekers craving depth, intention, and personal rebirth, and the "
            "journey is about rediscovery rather than rushing toward skills or outcomes."
        ),
        "ad_theme": "self_expression",
    },
    "narrator_and_lily": {
        "ad_name": "Heartfelt Connection",
        "ad_description": (
            "A warm, family-oriented storytelling ad centered on connection, presence, and shared creative "
            "moments. It appeals to parents, sentimental adults, and nostalgic creatives who value bonding "
            "and meaningful experiences. Creativity is framed as a way to nurture relationships and build memories."
        ),
        "ad_theme": "belonging",
    },
    "money_making": {
        "ad_name": "Creative Income Builder",
        "ad_description": (
            "This ad targets financially motivated viewers seeking side income, financial relief, or "
            "entrepreneurial creative opportunities. It sells the transformation from feeling financially "
            "stuck to earning through creative output. The emotional driver is empowerment via financial "
            "autonomy, with emphasis on ROI, sellable skills, and practical steps to turn skills into income."
        ),
        "ad_theme": "ambition",
    },
}
# ---------------------------------------------------------------------------
# Ensure connection_tone block exists with default emo fields
# ---------------------------------------------------------------------------
def ensure_tone_block(state: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee that stage_meta.connection_tone exists with default emo fields."""
    stage_meta = dict(state.get("stage_meta", {}) or {})

    tone_block = dict(stage_meta.get("connection_tone", {}) or {})
    tone_meta = dict(tone_block.get("metadata", {}) or {})
    tone_state = dict(tone_block.get("state", {}) or {})

    if "emo_tone_type" not in tone_meta:
        tone_meta["emo_tone_type"] = ""          # not decided yet
    if "confirm_tone" not in tone_meta:
        tone_meta["confirm_tone"] = "unclear"    # tone not confirmed yet
    if "emo_tone" not in tone_meta:
        tone_meta["emo_tone"] = "unclear"        # classification not set yet
    if "behavioral_signal" not in tone_meta:
        tone_meta["behavioral_signal"] = "mixed" # safe neutral default

    # These two will get overwritten by the tone agent when it copies intent,
    # but we initialize them so the structure is always present.
    tone_meta.setdefault("intent_type", "")
    tone_meta.setdefault("confirm_intent", "unclear")

    # --- State defaults ---
    tone_state.setdefault("last_theme", "")
    tone_state.setdefault("last_level", "")

    # Add new emotional tone tracking fields <-- ADD THESE LINES
    tone_state.setdefault("emo_tone_1", "")
    tone_state.setdefault("emo_tone_2", "")
    tone_state.setdefault("emo_tone_3", "")
    # turn = 0 means: tone agent has not asked any questions yet
    if "turn" not in tone_state:
        tone_state["turn"] = 1

    tone_block["metadata"] = tone_meta
    tone_block["state"] = tone_state
    stage_meta["connection_tone"] = tone_block

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
    - intent_mapping: intent categories for each option (for connection_intent agent).
    - metadata: agent-specific output (intent, motivation_type, barriers, persona, etc.).
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

    intent_mapping: List[str] = PydanticV1Field(
        ...,  # Required field - NO default!
        description=(
            "Exactly 4 intent categories matching the 4 options. "
            "Each value must be one of: 'self_expression', 'wellness', 'skill_growth', 'ambition', 'belonging', 'unsure'. "
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

    intent_mapping: List[str] = PydanticV1Field(  # ← Changed from List[Literal[...]]
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
    """Strict response format for emotional_tone agent only."""

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

    options: List[str] = PydanticV1Field(
        ...,  # Required field
        description="Exactly 4 answer options",
        min_items=4,
        max_items=4
    )

    emotional_mapping: List[str] = PydanticV1Field(
        ...,  # Required field - NO default!
        description=(
            "Exactly 4 emotional tone categories matching the 4 options. "
            "Each value must be one of: 'positive', 'neutral', 'tense', 'resistant', 'unsure', 'mixed'. "
            "Example: ['positive', 'neutral', 'tense', 'unsure']"
        ),
        min_items=4,
        max_items=4
    )

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

    motivation_mapping: List[str] = PydanticV1Field(
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

    barriers_mapping: List[str] = PydanticV1Field(
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

def get_user_chosen_intent(user_input: str, options: List[str], intent_mapping: List[str]) -> str:
    """
    Determine which intent the user chose based on their input.

    Args:
        user_input: The user's message (e.g., "A" or "Build new skills")
        options: List of option texts from previous turn
        intent_mapping: List of intent categories matching each option

    Returns:
        The intent category (e.g., "skill_growth", "wellness", "unsure")
    """
    if not options or not intent_mapping:
        return "unsure"

    user_input_clean = user_input.strip().lower()

    # Check if user typed A, B, C, D
    if len(user_input_clean) == 1 and user_input_clean in ['a', 'b', 'c', 'd']:
        idx = ord(user_input_clean) - ord('a')
        if 0 <= idx < len(intent_mapping):
            return intent_mapping[idx]

    # Check if user typed the full option text
    for i, option in enumerate(options):
        if option.lower() in user_input_clean or user_input_clean in option.lower():
            if i < len(intent_mapping):
                return intent_mapping[i]

    # Fallback
    return "unsure"


def compute_question_direction(theme_1: str, theme_2: str, current_turn: int) -> str:
    """
    Determine if next question should be 'broad' or 'deep'.

    Logic:
    - Turn 1: Always broad (exploring from ad)
    - Turn 2: Deep if theme_1 is clear and not unsure, broad otherwise
    - Turn 3: Deep if theme_1 == theme_2 (consistent), broad if mixed or unsure
    """
    if current_turn == 1:
        return "broad"

    if current_turn == 2:
        # We have theme_1 now
        if theme_1 in ["unsure", ""]:
            return "broad"
        else:
            return "deep"  # User showed clear intent in Q1

    if current_turn == 3:
        # We have theme_1 and theme_2
        if theme_1 == theme_2 and theme_1 not in ["unsure", ""]:
            return "deep"  # Consistent intent
        else:
            return "broad"  # Mixed or unsure

    return "broad"  # Fallback


def should_finalize_intent(theme_1: str, theme_2: str, theme_3: str, ad_theme: str) -> Tuple[bool, str]:
    """
    Finalize ONLY after all 3 intent questions have been answered.
    Majority vote:
      - If any two match → that theme
      - If all three different → 'mixed'
    """

    # --- CASE 1: Haven't reached question 3 yet ---
    if not theme_3 or theme_3 == "":
        return False, ""

    # --- CASE 2: We HAVE 3 answers, finalize ALWAYS ---
    # Majority logic
    if theme_1 == theme_2:
        return True, theme_1
    elif theme_1 == theme_3:
        return True, theme_1
    elif theme_2 == theme_3:
        return True, theme_2
    else:
        return True, "mixed"


def get_user_chosen_emotional_tone(user_input: str, options: List[str], emotional_mapping: List[str]) -> str:
    """
    Determine which emotional tone the user chose based on their input.

    Args:
        user_input: The user's answer text
        options: List of 4 option texts
        emotional_mapping: List of 4 emotional tone categories corresponding to options

    Returns:
        The emotional tone category the user selected
    """
    if not emotional_mapping or len(emotional_mapping) != 4:
        print(f"WARNING - get_user_chosen_emotional_tone: invalid emotional_mapping: {emotional_mapping}")
        return "unsure"

    if not options or len(options) != 4:
        print(f"WARNING - get_user_chosen_emotional_tone: invalid options: {options}")
        return "unsure"

    user_lower = user_input.lower().strip()

    # Try to match user input to one of the options
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        # Exact match or significant overlap
        if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
            return emotional_mapping[i]

    # Fallback: couldn't determine
    print(f"WARNING - get_user_chosen_emotional_tone: couldn't match '{user_input}' to options")
    return "unsure"


def compute_final_emotional_tone(emo_tone_1: str, emo_tone_2: str) -> str:
    """
    Determine the final emotional tone from 2 answers.

    Logic:
    1. If both answers are the same tone → use that tone
    2. If one is "unsure" → use the other tone
    3. If both are "unsure" → final tone is "unsure"
    4. If both are different (and neither is unsure) → final tone is "mixed"

    Args:
        emo_tone_1: Emotional tone from Q1
        emo_tone_2: Emotional tone from Q2

    Returns:
        Final emotional tone category
    """
    # Normalize inputs
    tone_1 = (emo_tone_1 or "").lower().strip()
    tone_2 = (emo_tone_2 or "").lower().strip()

    # If either is missing, return unsure
    if not tone_1 or not tone_2:
        return "unsure"

    # RULE 1: Both answers are the same → use that tone
    if tone_1 == tone_2:
        return tone_1

    # RULE 2: One is "unsure" → use the other tone
    if tone_1 == "unsure" and tone_2 != "unsure":
        return tone_2
    if tone_2 == "unsure" and tone_1 != "unsure":
        return tone_1

    # RULE 3: Both are "unsure" → final is "unsure"
    if tone_1 == "unsure" and tone_2 == "unsure":
        return "unsure"

    # RULE 4: Both are different (neither is unsure) → "mixed"
    return "mixed"


def should_finalize_emotional_tone(emo_tone_1: str, emo_tone_2: str) -> Tuple[bool, str]:
    """
    Determine if emotional tone assessment is complete and what the final tone is.

    Args:
        emo_tone_1: User's emotional tone from Q1
        emo_tone_2: User's emotional tone from Q2

    Returns:
        Tuple of (should_finalize, final_tone)
        - should_finalize: True if we have 2 answers
        - final_tone: The determined emotional tone
    """
    # Need both answers to finalize
    if not emo_tone_1 or not emo_tone_2:
        return (False, "")

    # We have 2 answers, time to finalize
    # Call the logic function to determine final tone
    final_tone = compute_final_emotional_tone(emo_tone_1, emo_tone_2)

    return (True, final_tone)


def update_intent_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update intent metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored intent_mapping
    2. Updates theme_1, theme_2, or theme_3 based on current turn
    3. Checks if intent should be finalized
    4. Updates confirm_intent and intent_type
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    intent_block = stage_meta.get("connection_intent", {}) or {}
    intent_state = dict(intent_block.get("state", {}) or {})
    intent_meta = dict(intent_block.get("metadata", {}) or {})

    # Get stored mapping from previous agent run
    prev_intent_mapping = intent_state.get("intent_mapping", [])
    prev_options = intent_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = intent_state.get("turn", 0)

    # Only process if we have a mapping and the user actually answered
    if not prev_intent_mapping or not user_answer:
        return state

    # Classify the user's answer
    chosen_intent = get_user_chosen_intent(user_answer, prev_options, prev_intent_mapping)

    print(
        f"DEBUG - update_intent_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_intent='{chosen_intent}'")

    # Update theme based on which question they just answered
    if current_turn == 1:
        # They just answered Q1
        intent_state["theme_1"] = chosen_intent
        intent_state["last_theme"] = chosen_intent
    elif current_turn == 2:
        # They just answered Q2
        intent_state["theme_2"] = chosen_intent
        intent_state["last_theme"] = chosen_intent
    elif current_turn == 3:
        # They just answered Q3
        intent_state["theme_3"] = chosen_intent
        intent_state["last_theme"] = chosen_intent

    # Get all three themes
    theme_1 = intent_state.get("theme_1", "")
    theme_2 = intent_state.get("theme_2", "")
    theme_3 = intent_state.get("theme_3", "")

    # Get ad_theme for finalization logic
    ad_data = state.get("ad_data", {}) or {}
    ad_theme = ad_data.get("ad_theme", "")

    # Check if we should finalize (after 3 questions)
    should_finalize, intent_type = should_finalize_intent(theme_1, theme_2, theme_3, ad_theme)

    print(f"DEBUG - should_finalize={should_finalize}, intent_type='{intent_type}'")

    if should_finalize:
        # We're done with intent questions
        intent_meta["confirm_intent"] = "clear"
        intent_meta["intent_type"] = intent_type
    else:
        # Keep asking questions
        intent_meta["confirm_intent"] = "unclear"
        intent_meta["intent_type"] = ""

    # Write back the updated metadata and state
    stage_meta["connection_intent"] = {
        "metadata": intent_meta,
        "state": intent_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }
def update_emotional_tone_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update emotional tone metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored emotional_mapping
    2. Updates emo_tone_1 or emo_tone_2 based on current turn
    3. Checks if emotional tone should be finalized
    4. Updates confirm_tone and emo_tone
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    tone_block = stage_meta.get("emotional_tone", {}) or {}
    tone_state = dict(tone_block.get("state", {}) or {})
    tone_meta = dict(tone_block.get("metadata", {}) or {})

    # Get stored mapping from previous agent run
    prev_emotional_mapping = tone_state.get("emotional_mapping", [])
    prev_options = tone_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = tone_state.get("turn", 0)

    # Only process if we have a mapping and the user actually answered
    if not prev_emotional_mapping or not user_answer:
        return state

    # Classify the user's answer
    chosen_tone = get_user_chosen_emotional_tone(user_answer, prev_options, prev_emotional_mapping)

    print(
        f"DEBUG - update_emotional_tone_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_tone='{chosen_tone}'")

    # Update emo_tone based on which question they just answered
    if current_turn == 1:
        # They just answered Q1
        tone_state["emo_tone_1"] = chosen_tone
    elif current_turn == 2:
        # They just answered Q2
        tone_state["emo_tone_2"] = chosen_tone

    # Get both tones
    emo_tone_1 = tone_state.get("emo_tone_1", "")
    emo_tone_2 = tone_state.get("emo_tone_2", "")

    # Check if we should finalize (after 2 questions)
    should_finalize, final_tone = should_finalize_emotional_tone(emo_tone_1, emo_tone_2)

    print(f"DEBUG - should_finalize={should_finalize}, final_tone='{final_tone}'")

    if should_finalize:
        # We're done with emotional tone questions
        tone_meta["confirm_tone"] = "clear"
        tone_meta["emo_tone"] = final_tone
    else:
        # Keep asking questions
        tone_meta["confirm_tone"] = "unclear"
        tone_meta["emo_tone"] = "unclear"

    # Write back the updated metadata and state
    stage_meta["emotional_tone"] = {
        "metadata": tone_meta,
        "state": tone_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }
def get_user_chosen_motivation(user_input: str, options: List[str], motivation_mapping: List[str]) -> str:
    """
    Determine which motivation the user chose based on their input.

    Args:
        user_input: The user's answer text
        options: List of 4 option texts
        motivation_mapping: List of 4 motivation categories corresponding to options

    Returns:
        The motivation category the user selected
    """
    if not motivation_mapping or len(motivation_mapping) != 4:
        print(f"WARNING - get_user_chosen_motivation: invalid motivation_mapping: {motivation_mapping}")
        return "unsure"

    if not options or len(options) != 4:
        print(f"WARNING - get_user_chosen_motivation: invalid options: {options}")
        return "unsure"

    user_lower = user_input.lower().strip()

    # Try to match user input to one of the options
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        # Exact match or significant overlap
        if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
            return motivation_mapping[i]

    # Fallback: couldn't determine
    print(f"WARNING - get_user_chosen_motivation: couldn't match '{user_input}' to options")
    return "unsure"


def compute_question_direction_motivation(motivation_1: str, motivation_2: str, motivation_3: str,
                                          current_turn: int) -> str:
    """
    Determine if next motivation question should be 'broad' or 'deep'.

    Logic (same as intent agent):
    - Turn 1: Always broad (exploring from intent)
    - Turn 2: Deep if motivation_1 is clear and not unsure, broad otherwise
    - Turn 3: Deep if motivation_1 == motivation_2 (consistent), broad if mixed or unsure
    - Turn 4: Deep if consistent pattern, broad if mixed

    Args:
        motivation_1: User's motivation from Q1
        motivation_2: User's motivation from Q2
        motivation_3: User's motivation from Q3
        current_turn: The question number we're about to ask (1-4)

    Returns:
        "broad" or "deep"
    """
    if current_turn == 1:
        return "broad"

    if current_turn == 2:
        # We have motivation_1 now
        if motivation_1 in ["unsure", ""]:
            return "broad"
        else:
            return "deep"  # User showed clear motivation in Q1

    if current_turn == 3:
        # We have motivation_1 and motivation_2
        if motivation_1 == motivation_2 and motivation_1 not in ["unsure", ""]:
            return "deep"  # Consistent motivation
        else:
            return "broad"  # Mixed or unsure

    if current_turn == 4:
        # We have motivation_1, motivation_2, and motivation_3
        # Check for consistency
        motivations = [motivation_1, motivation_2, motivation_3]
        # Remove unsure/empty
        clear_motivations = [m for m in motivations if m and m != "unsure"]

        if len(clear_motivations) >= 2:
            # Check if at least 2 are the same
            if clear_motivations[0] == clear_motivations[-1] or (
                    len(clear_motivations) >= 2 and clear_motivations[0] == clear_motivations[1]):
                return "deep"

        return "broad"

    return "broad"  # Fallback


def compute_final_motivation(motivation_1: str, motivation_2: str, motivation_3: str, motivation_4: str,
                             intent_type: str) -> str:
    """
    Determine the final motivation from 4 answers.

    Logic:
    1. If 2+ answers are "unsure" → "unsure"
    2. If same motivation appears 2+ times → that motivation
    3. If tie (2 motivations with 2 votes each) → use most recent
    4. If all different → "mixed"

    Args:
        motivation_1: Motivation from Q1
        motivation_2: Motivation from Q2
        motivation_3: Motivation from Q3
        motivation_4: Motivation from Q4
        intent_type: User's intent (used as tiebreaker if needed)

    Returns:
        Final motivation category
    """
    # Normalize inputs
    motivations = [
        (motivation_1 or "").lower().strip(),
        (motivation_2 or "").lower().strip(),
        (motivation_3 or "").lower().strip(),
        (motivation_4 or "").lower().strip()
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

    # Include intent_type as a tiebreaker option
    all_values = motivations + [intent_type.lower().strip()] if intent_type else motivations

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
        # Priority order: motivation_4 > motivation_3 > motivation_2 > motivation_1
        priority_order = [motivations[3], motivations[2], motivations[1], motivations[0]]
        for p in priority_order:
            if p in candidates:
                return p

        # Fallback
        return candidates[0]

    # RULE 3: All different → "mixed"
    return "mixed"
def should_finalize_motivation(motivation_1: str, motivation_2: str, motivation_3: str, motivation_4: str, intent_type: str) -> Tuple[bool, str]:
    """
    Determine if motivation assessment is complete and what the final motivation is.

    Args:
        motivation_1: User's motivation from Q1
        motivation_2: User's motivation from Q2
        motivation_3: User's motivation from Q3
        motivation_4: User's motivation from Q4
        intent_type: User's intent type (for tiebreaker)

    Returns:
        Tuple of (should_finalize, final_motivation)
        - should_finalize: True if we have 4 answers
        - final_motivation: The determined motivation type
    """
    # Need all 4 answers to finalize
    if not motivation_4 or motivation_4 == "":
        return (False, "")

    # We have 4 answers, time to finalize
    # Call the logic function to determine final motivation
    final_motivation = compute_final_motivation(motivation_1, motivation_2, motivation_3, motivation_4, intent_type)

    return (True, final_motivation)

def update_motivation_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update motivation metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored motivation_mapping
    2. Updates motivation_1, motivation_2, motivation_3, or motivation_4 based on current turn
    3. Updates last_motivation to most recent answer
    4. Checks if motivation should be finalized
    5. Updates confirm_motivation and motivation_type
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    motivation_block = stage_meta.get("motivation", {}) or {}
    motivation_state = dict(motivation_block.get("state", {}) or {})
    motivation_meta = dict(motivation_block.get("metadata", {}) or {})

    # Get stored mapping from previous agent run
    prev_motivation_mapping = motivation_state.get("motivation_mapping", [])
    prev_options = motivation_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = motivation_state.get("turn", 0)

    # Only process if we have a mapping and the user actually answered
    if not prev_motivation_mapping or not user_answer:
        return state

    # Classify the user's answer
    chosen_motivation = get_user_chosen_motivation(user_answer, prev_options, prev_motivation_mapping)

    print(
        f"DEBUG - update_motivation_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_motivation='{chosen_motivation}'")

    # Update motivation based on which question they just answered
    if current_turn == 1:
        # They just answered Q1
        motivation_state["motivation_1"] = chosen_motivation
        motivation_state["last_motivation"] = chosen_motivation
    elif current_turn == 2:
        # They just answered Q2
        motivation_state["motivation_2"] = chosen_motivation
        motivation_state["last_motivation"] = chosen_motivation
    elif current_turn == 3:
        # They just answered Q3
        motivation_state["motivation_3"] = chosen_motivation
        motivation_state["last_motivation"] = chosen_motivation
    elif current_turn == 4:
        # They just answered Q4
        motivation_state["motivation_4"] = chosen_motivation
        motivation_state["last_motivation"] = chosen_motivation

    # Get all four motivations
    motivation_1 = motivation_state.get("motivation_1", "")
    motivation_2 = motivation_state.get("motivation_2", "")
    motivation_3 = motivation_state.get("motivation_3", "")
    motivation_4 = motivation_state.get("motivation_4", "")

    # Get intent_type for finalization logic
    intent_block = stage_meta.get("connection_intent", {}) or {}
    intent_meta = intent_block.get("metadata", {}) or {}
    intent_type = intent_meta.get("intent_type", "")

    # Check if we should finalize (after 4 questions)
    should_finalize, final_motivation = should_finalize_motivation(
        motivation_1, motivation_2, motivation_3, motivation_4, intent_type
    )

    print(f"DEBUG - should_finalize={should_finalize}, final_motivation='{final_motivation}'")

    if should_finalize:
        # We're done with motivation questions
        motivation_meta["confirm_motivation"] = "clear"
        motivation_meta["motivation_type"] = final_motivation
    else:
        # Keep asking questions
        motivation_meta["confirm_motivation"] = "unclear"
        motivation_meta["motivation_type"] = ""

    # Write back the updated metadata and state
    stage_meta["motivation"] = {
        "metadata": motivation_meta,
        "state": motivation_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }
def get_user_chosen_barrier(user_input: str, options: List[str], barriers_mapping: List[str]) -> str:
    """
    Determine which barrier the user chose based on their input.

    Args:
        user_input: The user's answer text
        options: List of 4 option texts
        barriers_mapping: List of 4 barrier categories corresponding to options

    Returns:
        The barrier category the user selected
    """
    if not barriers_mapping or len(barriers_mapping) != 4:
        print(f"WARNING - get_user_chosen_barrier: invalid barriers_mapping: {barriers_mapping}")
        return "unsure"

    if not options or len(options) != 4:
        print(f"WARNING - get_user_chosen_barrier: invalid options: {options}")
        return "unsure"

    user_lower = user_input.lower().strip()

    # Try to match user input to one of the options
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        # Exact match or significant overlap
        if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
            return barriers_mapping[i]

    # Fallback: couldn't determine
    print(f"WARNING - get_user_chosen_barrier: couldn't match '{user_input}' to options")
    return "unsure"

def compute_question_direction_barriers(barrier_1: str, barrier_2: str, barrier_3: str,
                                        current_turn: int) -> str:
    """
    Determine if next barrier question should be 'broad' or 'deep'.

    Logic (same as motivation and intent agents):
    - Turn 1: Always broad (exploring from motivation)
    - Turn 2: Deep if barrier_1 is clear and not unsure, broad otherwise
    - Turn 3: Deep if barrier_1 == barrier_2 (consistent), broad if mixed or unsure
    - Turn 4: Deep if consistent pattern, broad if mixed

    Args:
        barrier_1: User's barrier from Q1
        barrier_2: User's barrier from Q2
        barrier_3: User's barrier from Q3
        current_turn: The question number we're about to ask (1-4)

    Returns:
        "broad" or "deep"
    """
    if current_turn == 1:
        return "broad"

    if current_turn == 2:
        # We have barrier_1 now
        if barrier_1 in ["unsure", ""]:
            return "broad"
        else:
            return "deep"  # User showed clear barrier in Q1

    if current_turn == 3:
        # We have barrier_1 and barrier_2
        if barrier_1 == barrier_2 and barrier_1 not in ["unsure", ""]:
            return "deep"  # Consistent barrier
        else:
            return "broad"  # Mixed or unsure

    if current_turn == 4:
        # We have barrier_1, barrier_2, and barrier_3
        # Check for consistency
        barriers = [barrier_1, barrier_2, barrier_3]
        # Remove unsure/empty
        clear_barriers = [b for b in barriers if b and b != "unsure"]

        if len(clear_barriers) >= 2:
            # Check if at least 2 are the same
            if clear_barriers[0] == clear_barriers[-1] or (
                    len(clear_barriers) >= 2 and clear_barriers[0] == clear_barriers[1]):
                return "deep"

        return "broad"

    return "broad"  # Fallback

def compute_final_barrier(barrier_1: str, barrier_2: str, barrier_3: str, barrier_4: str,
                         motivation_type: str) -> str:
    """
    Determine the final barrier from 4 answers.

    Logic:
    1. If 2+ answers are "unsure" → "unsure"
    2. If same barrier appears 2+ times → that barrier
    3. If tie (2 barriers with 2 votes each) → use most recent
    4. If all different → "mixed"

    Args:
        barrier_1: Barrier from Q1
        barrier_2: Barrier from Q2
        barrier_3: Barrier from Q3
        barrier_4: Barrier from Q4
        motivation_type: User's motivation (used as tiebreaker if needed)

    Returns:
        Final barrier category
    """
    # Normalize inputs
    barriers = [
        (barrier_1 or "").lower().strip(),
        (barrier_2 or "").lower().strip(),
        (barrier_3 or "").lower().strip(),
        (barrier_4 or "").lower().strip()
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
        # Priority order: barrier_4 > barrier_3 > barrier_2 > barrier_1
        priority_order = [barriers[3], barriers[2], barriers[1], barriers[0]]
        for p in priority_order:
            if p in candidates:
                return p

        # Fallback
        return candidates[0]

    # RULE 3: All different → "mixed"
    return "mixed"

def should_finalize_barriers(barrier_1: str, barrier_2: str, barrier_3: str, barrier_4: str,
                            motivation_type: str) -> Tuple[bool, str]:
    """
    Determine if barriers assessment is complete and what the final barrier is.

    Args:
        barrier_1: User's barrier from Q1
        barrier_2: User's barrier from Q2
        barrier_3: User's barrier from Q3
        barrier_4: User's barrier from Q4
        motivation_type: User's motivation type (for tiebreaker)

    Returns:
        Tuple of (should_finalize, final_barrier)
        - should_finalize: True if we have 4 answers
        - final_barrier: The determined barrier type
    """
    # Need all 4 answers to finalize
    if not barrier_4 or barrier_4 == "":
        return (False, "")

    # We have 4 answers, time to finalize
    # Call the logic function to determine final barrier
    final_barrier = compute_final_barrier(barrier_1, barrier_2, barrier_3, barrier_4, motivation_type)

    return (True, final_barrier)
def update_barriers_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update barriers metadata after user answers, BEFORE the agent runs again.

    This function:
    1. Classifies the user's answer using stored barriers_mapping
    2. Updates barrier_1, barrier_2, barrier_3, or barrier_4 based on current turn
    3. Updates last_barrier to most recent answer
    4. Checks if barriers should be finalized
    5. Updates confirm_barriers and barrier_type
    """
    import copy

    # Deep copy to avoid mutations
    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    barriers_block = stage_meta.get("barriers", {}) or {}
    barriers_state = dict(barriers_block.get("state", {}) or {})
    barriers_meta = dict(barriers_block.get("metadata", {}) or {})

    # Get stored mapping from previous agent run
    prev_barriers_mapping = barriers_state.get("barriers_mapping", [])
    prev_options = barriers_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = barriers_state.get("turn", 0)

    # Only process if we have a mapping and the user actually answered
    if not prev_barriers_mapping or not user_answer:
        return state

    # Classify the user's answer
    chosen_barrier = get_user_chosen_barrier(user_answer, prev_options, prev_barriers_mapping)

    print(
        f"DEBUG - update_barriers_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_barrier='{chosen_barrier}'")

    # Update barrier based on which question they just answered
    if current_turn == 1:
        # They just answered Q1
        barriers_state["barrier_1"] = chosen_barrier
        barriers_state["last_barrier"] = chosen_barrier
    elif current_turn == 2:
        # They just answered Q2
        barriers_state["barrier_2"] = chosen_barrier
        barriers_state["last_barrier"] = chosen_barrier
    elif current_turn == 3:
        # They just answered Q3
        barriers_state["barrier_3"] = chosen_barrier
        barriers_state["last_barrier"] = chosen_barrier
    elif current_turn == 4:
        # They just answered Q4
        barriers_state["barrier_4"] = chosen_barrier
        barriers_state["last_barrier"] = chosen_barrier

    # Get all four barriers
    barrier_1 = barriers_state.get("barrier_1", "")
    barrier_2 = barriers_state.get("barrier_2", "")
    barrier_3 = barriers_state.get("barrier_3", "")
    barrier_4 = barriers_state.get("barrier_4", "")

    # Get motivation_type for finalization logic
    motivation_block = stage_meta.get("motivation", {}) or {}
    motivation_meta = motivation_block.get("metadata", {}) or {}
    motivation_type = motivation_meta.get("motivation_type", "")

    # Check if we should finalize (after 4 questions)
    should_finalize, final_barrier = should_finalize_barriers(
        barrier_1, barrier_2, barrier_3, barrier_4, motivation_type
    )

    print(f"DEBUG - should_finalize={should_finalize}, final_barrier='{final_barrier}'")

    if should_finalize:
        # We're done with barriers questions
        barriers_meta["confirm_barriers"] = "clear"
        barriers_meta["barrier_type"] = final_barrier
    else:
        # Keep asking questions
        barriers_meta["confirm_barriers"] = "unclear"
        barriers_meta["barrier_type"] = ""

    # Write back the updated metadata and state
    stage_meta["barriers"] = {
        "metadata": barriers_meta,
        "state": barriers_state,
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
      - intent_type: one of "unsure", "self_expression", "wellness",
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
        my_block = stage_meta.get(self.info_type, {}) or {}
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

        # FOR CONNECTION_INTENT AGENT: compute last_intent and question_direction
        if self.info_type == "connection_intent":
            stage_meta = state.get("stage_meta", {}) or {}
            intent_block = stage_meta.get("connection_intent", {}) or {}
            intent_state = intent_block.get("state", {}) or {}

            current_turn = intent_state.get("turn", 0)
            ad_data = state.get("ad_data", {}) or {}
            ad_theme = ad_data.get("ad_theme", "")

            theme_1 = intent_state.get("theme_1", "")
            theme_2 = intent_state.get("theme_2", "")
            last_theme = intent_state.get("last_theme", "")

            # Determine last_intent
            if current_turn == 0:
                # First turn - use ad_theme
                last_intent = ad_theme
            else:
                # Use last_theme if available, otherwise ad_theme
                last_intent = last_theme if last_theme else ad_theme

            # Compute question_direction
            question_direction = compute_question_direction(theme_1, theme_2, current_turn + 1)

            ctx["last_intent"] = last_intent
            ctx["question_direction"] = question_direction

        # FOR EMOTIONAL_TONE AGENT: get last_intent from connection_intent results
        if self.info_type == "emotional_tone":
            stage_meta = state.get("stage_meta", {}) or {}

            # Get the finalized intent from connection_intent agent
            intent_block = stage_meta.get("connection_intent", {}) or {}
            intent_meta = intent_block.get("metadata", {}) or {}
            intent_state = intent_block.get("state", {}) or {}

            # Use the finalized intent_type, or fall back to last_theme, or ad_theme
            intent_type = intent_meta.get("intent_type", "")
            last_theme = intent_state.get("last_theme", "")
            ad_data = state.get("ad_data", {}) or {}
            ad_theme = ad_data.get("ad_theme", "")

            # Determine last_intent for emotional tone questions
            if intent_type:
                last_intent = intent_type  # Use finalized intent
            elif last_theme:
                last_intent = last_theme
            else:
                last_intent = ad_theme

            ctx["last_intent"] = last_intent

        # FOR MOTIVATION AGENT: compute last_motivation and question_direction
        if self.info_type == "motivation":
            stage_meta = state.get("stage_meta", {}) or {}

            # Get the finalized intent from connection_intent agent
            intent_block = stage_meta.get("connection_intent", {}) or {}
            intent_meta = intent_block.get("metadata", {}) or {}
            intent_type = intent_meta.get("intent_type", "")

            # Get motivation state
            motivation_block = stage_meta.get("motivation", {}) or {}
            motivation_state = motivation_block.get("state", {}) or {}

            current_turn = motivation_state.get("turn", 0)

            motivation_1 = motivation_state.get("motivation_1", "")
            motivation_2 = motivation_state.get("motivation_2", "")
            motivation_3 = motivation_state.get("motivation_3", "")
            last_motivation_state = motivation_state.get("last_motivation", "")

            # Determine last_motivation
            if current_turn == 0:
                # First turn - use intent_type from intent agent
                last_motivation = intent_type if intent_type else ""
            else:
                # Use last_motivation if available, otherwise intent_type
                last_motivation = last_motivation_state if last_motivation_state else intent_type

            # Compute question_direction
            question_direction = compute_question_direction_motivation(
                motivation_1, motivation_2, motivation_3, current_turn + 1
            )

            ctx["last_motivation"] = last_motivation
            ctx["question_direction"] = question_direction

            # FOR BARRIERS AGENT: compute last_barrier and question_direction
        if self.info_type == "barriers":
            stage_meta = state.get("stage_meta", {}) or {}

            # Get the finalized motivation from motivation agent
            motivation_block = stage_meta.get("motivation", {}) or {}
            motivation_meta = motivation_block.get("metadata", {}) or {}
            motivation_type = motivation_meta.get("motivation_type", "")

            # Get barriers state
            barriers_block = stage_meta.get("barriers", {}) or {}
            barriers_state = barriers_block.get("state", {}) or {}

            current_turn = barriers_state.get("turn", 0)

            barrier_1 = barriers_state.get("barrier_1", "")
            barrier_2 = barriers_state.get("barrier_2", "")
            barrier_3 = barriers_state.get("barrier_3", "")
            last_barrier_state = barriers_state.get("last_barrier", "")

            # Determine last_barrier
            if current_turn == 0:
                # First turn - use motivation_type from motivation agent
                last_barrier = motivation_type if motivation_type else ""
            else:
                # Use last_barrier if available, otherwise motivation_type
                last_barrier = last_barrier_state if last_barrier_state else motivation_type

            # Compute question_direction
            question_direction = compute_question_direction_barriers(
                barrier_1, barrier_2, barrier_3, current_turn + 1
            )

            ctx["last_barrier"] = last_barrier
            ctx["question_direction"] = question_direction
            # FOR SUMMARY AGENT: gather all insights from all 4 agents
        if self.info_type == "summary":
            print("DEBUG - Summary agent building context")
            stage_meta = state.get("stage_meta", {}) or {}
            print(f"DEBUG - stage_meta keys: {list(stage_meta.keys())}")
            # Get intent data
            intent_block = stage_meta.get("connection_intent", {}) or {}
            intent_meta = intent_block.get("metadata", {}) or {}
            intent_state = intent_block.get("state", {}) or {}

            # Get tone data
            tone_block = stage_meta.get("emotional_tone", {}) or {}
            tone_meta = tone_block.get("metadata", {}) or {}
            tone_state = tone_block.get("state", {}) or {}

            # Get motivation data
            motivation_block = stage_meta.get("motivation", {}) or {}
            motivation_meta = motivation_block.get("metadata", {}) or {}
            motivation_state = motivation_block.get("state", {}) or {}

            # Get barriers data
            barriers_block = stage_meta.get("barriers", {}) or {}
            barriers_meta = barriers_block.get("metadata", {}) or {}
            barriers_state = barriers_block.get("state", {}) or {}

            # Add intent data to context
            ctx["intent_type"] = intent_meta.get("intent_type", "")
            ctx["theme_1"] = intent_state.get("theme_1", "")
            ctx["theme_2"] = intent_state.get("theme_2", "")
            ctx["theme_3"] = intent_state.get("theme_3", "")

            # Add tone data to context
            ctx["emo_tone"] = tone_meta.get("emo_tone", "")
            ctx["emo_tone_1"] = tone_state.get("emo_tone_1", "")
            ctx["emo_tone_2"] = tone_state.get("emo_tone_2", "")

            # Add motivation data to context
            ctx["motivation_type"] = motivation_meta.get("motivation_type", "")
            ctx["motivation_1"] = motivation_state.get("motivation_1", "")
            ctx["motivation_2"] = motivation_state.get("motivation_2", "")
            ctx["motivation_3"] = motivation_state.get("motivation_3", "")
            ctx["motivation_4"] = motivation_state.get("motivation_4", "")

            # Add barriers data to context
            ctx["barrier_type"] = barriers_meta.get("barrier_type", "")
            ctx["barrier_1"] = barriers_state.get("barrier_1", "")
            ctx["barrier_2"] = barriers_state.get("barrier_2", "")
            ctx["barrier_3"] = barriers_state.get("barrier_3", "")
            ctx["barrier_4"] = barriers_state.get("barrier_4", "")

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
                intent_mapping=["", "", "", ""],
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
            if self.info_type == "connection_intent":
                structured_llm = llm.with_structured_output(ConnectionIntentResponse)
                strict_response: ConnectionIntentResponse = structured_llm.invoke(msgs)
                # Convert to AgentResponse format
                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=strict_response.options,
                    intent_mapping=strict_response.intent_mapping,
                    metadata={},
                    state={}
                )
            # Use strict schema for emotional_tone agent
            elif self.info_type == "emotional_tone":
                structured_llm = llm.with_structured_output(EmotionalToneResponse)
                strict_response: EmotionalToneResponse = structured_llm.invoke(msgs)
                # Convert to AgentResponse format
                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=strict_response.options,
                    intent_mapping=strict_response.emotional_mapping,  # Map emotional_mapping to intent_mapping
                    metadata={},
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
                    intent_mapping=strict_response.motivation_mapping,  # Map motivation_mapping to intent_mapping
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
                    intent_mapping=strict_response.barriers_mapping,  # Map barriers_mapping to intent_mapping
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
                    intent_mapping=["", "", "", ""],  # Required field with 4 items
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
                    intent_mapping=["self_expression", "skill_growth", "wellness", "unsure"],
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
                    intent_mapping=["positive", "neutral", "tense", "unsure"],
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
                    intent_mapping=["progress", "expression", "relief", "unsure"],
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
                    intent_mapping=["consistency", "overwhelm", "time_energy", "unsure"],
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
                    intent_mapping=["", "", "", ""],
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
                    intent_mapping=["", "", "", ""],
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

        if response.affirmation:
            display_text = response.affirmation + "\n\n"

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
        old_block = stage_meta_prev.get(self.info_type) or {}
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
                new_state["last_theme"] = ad_theme  # Start with ad_theme

            # Set level based on turn
            if current_turn == 1:
                new_state["last_level"] = "L1"
            elif current_turn == 2:
                new_state["last_level"] = "L2"
            elif current_turn == 3:
                new_state["last_level"] = "L3"
            else:
                new_state["last_level"] = "L4"

            # Store intent_mapping and options for next turn's classification
            if response.intent_mapping:
                new_state["intent_mapping"] = response.intent_mapping
            if response.options:
                new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_intent_metadata_after_answer)
            new_meta.setdefault("confirm_intent", "unclear")
            new_meta.setdefault("intent_type", "")
            new_state.setdefault("last_theme", "")
            stage_meta_prev[self.info_type] = {
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
                new_state["emo_tone_1"] = ""
                new_state["emo_tone_2"] = ""

            # Store emotional_mapping and options from LLM response
            if response.intent_mapping:  # emotional_mapping was mapped to intent_mapping
                new_state["emotional_mapping"] = response.intent_mapping
            if response.options:
                new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_emotional_tone_metadata_after_answer)
            new_meta.setdefault("confirm_tone", "unclear")
            new_meta.setdefault("emo_tone", "unclear")

            stage_meta_prev[self.info_type] = {
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
                new_state["motivation_1"] = ""
                new_state["motivation_2"] = ""
                new_state["motivation_3"] = ""
                new_state["motivation_4"] = ""
                new_state["last_motivation"] = ""  # Will be set after first answer

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

            # Store motivation_mapping and options from LLM response
            if response.intent_mapping:  # motivation_mapping was mapped to intent_mapping
                new_state["motivation_mapping"] = response.intent_mapping
            if response.options:
                new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_motivation_metadata_after_answer)
            new_meta.setdefault("confirm_motivation", "unclear")
            new_meta.setdefault("motivation_type", "")
            new_state.setdefault("last_motivation", "")

            stage_meta_prev[self.info_type] = {
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
                new_state["barrier_1"] = ""
                new_state["barrier_2"] = ""
                new_state["barrier_3"] = ""
                new_state["barrier_4"] = ""
                new_state["last_barrier"] = ""  # Will be set after first answer

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

            # Store barriers_mapping and options from LLM response
            if response.intent_mapping:  # barriers_mapping was mapped to intent_mapping
                new_state["barriers_mapping"] = response.intent_mapping
            if response.options:
                new_state["options"] = response.options

            # Keep metadata fields (these will be updated by update_barriers_metadata_after_answer)
            new_meta.setdefault("confirm_barriers", "unclear")
            new_meta.setdefault("barrier_type", "")
            new_state.setdefault("last_barrier", "")

            stage_meta_prev[self.info_type] = {
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
            if hasattr(response, 'summary_text'):
                new_meta["summary_text"] = response.summary_text
                new_meta["recommendations"] = response.recommendations
            else:
                # Fallback: get from metadata
                new_meta["summary_text"] = response.metadata.get("summary_text", "")
                new_meta["recommendations"] = response.metadata.get("recommendations", [])


            stage_meta_prev[self.info_type] = {
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

            # Compute final intent_type deterministically
            final_intent_type = compute_final_intent_from_state(
                ci_state_block,
                ad_theme=ad_theme,
            )

            # Finalizer always sets confirm_intent = "clear"
            new_meta["intent_type"] = final_intent_type
            new_meta["confirm_intent"] = "clear"

            # Mark this as the final (TURN 4) snapshot
            new_state["turn"] = 4
            new_state["last_level"] = "L4"

            # Update last_theme if we have a clear non-unsure, non-mixed category
            if final_intent_type not in ("unsure", "mixed", ""):
                new_state["last_theme"] = final_intent_type
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
            #   - emo_tone_1 MUST be "" (no classification yet)
            #   - emo_tone_2 MUST be ""
            #
            # TURN 2→3 (Q2 phase):
            #   - emo_tone_1 may be filled by the LLM from Q1
            #   - emo_tone_2 MUST stay ""
            #
            # TURN 3→4 (FINAL phase):
            #   - LLM sets emo_tone_2 + final emo_tone / confirm_tone
            #
            # TURN 4+ (DONE phase):
            #   - Everything stays as-is

            if incoming_turn <= 1:
                # Q1 phase - first tone question → wipe any premature tone labels
                new_state["emo_tone_1"] = ""
                new_state["emo_tone_2"] = ""
                new_state.setdefault("emo_tone_3", "")
                # Ensure metadata fields exist
                new_meta.setdefault("emo_tone", "unclear")
                new_meta.setdefault("emo_tone_type", "")
                new_meta.setdefault("confirm_tone", "unclear")

            elif incoming_turn == 2:
                # Q2 phase - emo_tone_1 should already be set by LLM from Q1
                # Keep emo_tone_2 blank until Turn 3
                new_state.setdefault("emo_tone_1", "")  # keep it if already set
                new_state["emo_tone_2"] = ""  # not set yet
                new_state.setdefault("emo_tone_3", "")
                # Keep tone unclear during Q2
                new_meta.setdefault("emo_tone", "unclear")
                new_meta.setdefault("emo_tone_type", "")
                new_meta.setdefault("confirm_tone", "unclear")

            elif incoming_turn == 3:
                # FINAL phase - LLM should set emo_tone_2 + final classification
                new_state.setdefault("emo_tone_1", "")
                new_state.setdefault("emo_tone_2", "")  # LLM MUST populate this now
                new_state.setdefault("emo_tone_3", "")

                # Safety check: if LLM didn't set emo_tone_2, log warning
                if not new_state.get("emo_tone_2"):
                    print("WARNING: emo_tone_2 not set by LLM during FINAL phase")

                # Safety check: if LLM didn't set confirm_tone to "clear", force it
                if new_meta.get("confirm_tone") != "clear":
                    print("WARNING: LLM didn't set confirm_tone='clear' in FINAL phase, forcing it")
                    new_meta["confirm_tone"] = "clear"

                    # If LLM also didn't set final emo_tone, compute it as fallback
                    if not new_meta.get("emo_tone") or new_meta.get("emo_tone") == "unclear":
                        emo_1 = new_state.get("emo_tone_1", "")
                        emo_2 = new_state.get("emo_tone_2", "")

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
                        new_meta["emo_tone"] = final_tone
                        new_meta["emo_tone_type"] = final_tone

            else:
                # DONE phase (turn >= 4) - don't ask more questions
                new_state.setdefault("emo_tone_1", "")
                new_state.setdefault("emo_tone_2", "")
                new_state.setdefault("emo_tone_3", "")
                # Ensure confirm_tone stays "clear"
                if new_meta.get("confirm_tone") != "clear":
                    print("WARNING: confirm_tone not 'clear' in DONE phase, fixing")
                    new_meta["confirm_tone"] = "clear"

            # 3) Copy canonical intent from connection_intent into tone metadata
            intent_block = stage_meta_prev.get("connection_intent") or {}
            intent_meta = intent_block.get("metadata") or {}

            canonical_intent_type = intent_meta.get("intent_type")
            canonical_confirm_intent = intent_meta.get("confirm_intent")

            if canonical_intent_type is not None:
                new_meta["intent_type"] = canonical_intent_type
            if canonical_confirm_intent is not None:
                new_meta["confirm_intent"] = canonical_confirm_intent

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

        # --- Read INTENT block (connection_intent) ---
        intent_block = stage_meta.get("connection_intent", {}) or {}
        intent_meta = intent_block.get("metadata", {}) or {}

        # --- Read TONE block (connection_tone) ---
        tone_block = stage_meta.get("connection_tone", {}) or {}
        tone_meta = tone_block.get("metadata", {}) or {}

        # Status fields
        intent_status = (
                intent_meta.get("confirm_intent")
                or intent_meta.get("intent_status")
                or ""
        ).lower()
        intent_type = (intent_meta.get("intent_type") or "").lower()
        emo_tone = (
                tone_meta.get("emo_tone")
                or tone_meta.get("emotional_tone")
                or "unclear"
        ).lower()

        # Turn counts per agent (fallback to counts of collected answers)
        intent_turns = len(collected.get("connection_intent", []))
        tone_turns = len(collected.get("connection_tone", []))
        total_turns = intent_turns + tone_turns

        ad_data = state.get("ad_data", {}) or {}
        ad_theme = ad_data.get("ad_creative_theme", "").lower()

        print(
            f"DEBUG SUPERVISOR (fallback) - intent_turns: {intent_turns}, "
            f"tone_turns: {tone_turns}, total_turns: {total_turns}, "
            f"intent_status: {intent_status}, emo_tone: {emo_tone}, "
            f"ad_theme: {ad_theme}, intent_type: {intent_type}"
        )

        # Stage 1 exit readiness (roughly mirror YAML logic)
        ready = (intent_status in ("clear", "unsure")) and (emo_tone not in ("resistant", "unclear"))

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

            # Prefer explicit intent_type, fall back to ad theme
            route_key = intent_type if intent_type in routing_map else ad_theme

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
        if intent_status in ("", "unclear"):
            if "connection_intent" in self.agent_keys:
                print("DEBUG SUPERVISOR (fallback) - Continuing with connection_intent agent")
                return {
                    "agent_index": self.agent_keys.index("connection_intent"),
                    "next_agent": "connection_intent",
                    "exchanges_with_current": 0,
                }

        # 2) If intent is set but tone is unclear → move to tone agent
        if emo_tone in ("", "unclear"):
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

        # Extract intent info
        intent_block = stage_meta.get("connection_intent", {}) or {}
        intent_meta = intent_block.get("metadata", {}) or {}
        intent_state = intent_block.get("state", {}) or {}

        intent_status = intent_meta.get("confirm_intent", "unclear")
        intent_turn = intent_state.get("turn", 0)

        # Extract tone info - try emotional_tone first, fallback to connection_tone
        tone_block = stage_meta.get("emotional_tone", {}) or stage_meta.get("connection_tone", {}) or {}
        tone_meta = tone_block.get("metadata", {}) or {}
        tone_state = tone_block.get("state", {}) or {}

        confirm_tone = tone_meta.get("confirm_tone", "unclear")
        tone_turn = tone_state.get("turn", 0)
        emo_tone_1 = tone_state.get("emo_tone_1", "")
        emo_tone_2 = tone_state.get("emo_tone_2", "")
        emo_tone = tone_meta.get("emo_tone", "unclear")

        # Extract motivation info
        motivation_block = stage_meta.get("motivation", {}) or {}
        motivation_meta = motivation_block.get("metadata", {}) or {}
        motivation_state = motivation_block.get("state", {}) or {}

        confirm_motivation = motivation_meta.get("confirm_motivation", "unclear")
        motivation_turn = motivation_state.get("turn", 0)
        motivation_1 = motivation_state.get("motivation_1", "")
        motivation_2 = motivation_state.get("motivation_2", "")
        motivation_3 = motivation_state.get("motivation_3", "")
        motivation_4 = motivation_state.get("motivation_4", "")
        motivation_type = motivation_meta.get("motivation_type", "unclear")

        # Extract barriers info
        barriers_block = stage_meta.get("barriers", {}) or {}
        barriers_meta = barriers_block.get("metadata", {}) or {}
        barriers_state = barriers_block.get("state", {}) or {}

        confirm_barriers = barriers_meta.get("confirm_barriers", "unclear")
        barriers_turn = barriers_state.get("turn", 0)
        barrier_1 = barriers_state.get("barrier_1", "")
        barrier_2 = barriers_state.get("barrier_2", "")
        barrier_3 = barriers_state.get("barrier_3", "")
        barrier_4 = barriers_state.get("barrier_4", "")
        barrier_type = barriers_meta.get("barrier_type", "unclear")

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

            # Intent variables
            "intent_status": intent_status,
            "intent_turn": str(intent_turn),

            # Tone variables
            "confirm_tone": confirm_tone,
            "tone_turn": str(tone_turn),
            "emo_tone_1": emo_tone_1,
            "emo_tone_2": emo_tone_2,
            "emo_tone": emo_tone,

            # Motivation variables
            "confirm_motivation": confirm_motivation,
            "motivation_turn": str(motivation_turn),
            "motivation_1": motivation_1,
            "motivation_2": motivation_2,
            "motivation_3": motivation_3,
            "motivation_4": motivation_4,
            "motivation_type": motivation_type,

            # Barriers variables
            "confirm_barriers": confirm_barriers,
            "barriers_turn": str(barriers_turn),
            "barrier_1": barrier_1,
            "barrier_2": barrier_2,
            "barrier_3": barrier_3,
            "barrier_4": barrier_4,
            "barrier_type": barrier_type,

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
        tone_block = stage_meta.get("connection_tone", {}) or {}
        tone_meta = tone_block.get("metadata", {}) or {}

        # New tone fields you want to use
        tone_status = (tone_meta.get("confirm_tone") or "").lower()
        # tone_status: "" or "unclear" means tone not finalized

        # 🔒 HARD RULE:
        # If supervisor tries to FINISH but tone is not confirmed yet,
        # FORCE it to go to connection_tone instead (if that agent exists).
        if nxt in ("finish", "end") and tone_status in ("", "unclear"):
            if "connection_tone" in main_keys:
                print("DEBUG ROUTER - Overriding FINISH → connection_tone because confirm_tone is not clear.")
                return "connection_tone"

        # Normal FINISH handling (only when tone_status is clear or we don't care)
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
    state = ensure_tone_block(state)
    # Update intent metadata if user answered an intent question
    last_agent = state.get("last_agent", "")

    # Update intent metadata if user answered an intent question
    if last_agent == "connection_intent" and msg:
        state = update_intent_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After intent metadata update: confirm_intent={state.get('stage_meta', {}).get('connection_intent', {}).get('metadata', {}).get('confirm_intent')}")

    # Update emotional tone metadata if user answered an emotional tone question
    if last_agent == "emotional_tone" and msg:
        state = update_emotional_tone_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After emotional tone metadata update: confirm_tone={state.get('stage_meta', {}).get('emotional_tone', {}).get('metadata', {}).get('confirm_tone')}")
        # Update motivation metadata if user answered a motivation question
    if last_agent == "motivation" and msg:
        state = update_motivation_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After motivation metadata update: confirm_motivation={state.get('stage_meta', {}).get('motivation', {}).get('metadata', {}).get('confirm_motivation')}")
        # Update barriers metadata if user answered a barriers question
    if last_agent == "barriers" and msg:
        state = update_barriers_metadata_after_answer(state, msg)
        print(
            f"DEBUG - After barriers metadata update: confirm_barriers={state.get('stage_meta', {}).get('barriers', {}).get('metadata', {}).get('confirm_barriers')}")
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
            stage_meta={},  # we'll fill connection_tone inside ensure_tone_block
            last_options=[],
            ad_data=st.session_state.ad_data,
            user_profile={},
        )

        # ✅ make sure connection_tone + emo_tone_type + confirm_tone exist from the start
        base_state = ensure_tone_block(base_state)

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
        st.header("Collected Info")
        for k, v in st.session_state.state.get("collected_info", {}).items():
            st.markdown(f"**{k}** ({len(v)})")
            for i, it in enumerate(v[-5:]):
                st.write(f"- {i + 1}. {it}")

        st.markdown("---")
        st.header("Conversation Insights")
        meta = st.session_state.state.get("stage_meta", {}) or {}

        # Intent block (connection_intent)
        intent_block = meta.get("connection_intent", {}) or {}
        intent_meta = intent_block.get("metadata", {}) or {}
        intent_state = intent_block.get("state", {}) or {}

        # Tone block (emotional_tone) - CHECK BOTH emotional_tone AND connection_tone
        tone_block = meta.get("emotional_tone", {}) or meta.get("connection_tone", {}) or {}
        tone_meta = tone_block.get("metadata", {}) or {}
        tone_state = tone_block.get("state", {}) or {}

        # Motivation block
        motivation_block = meta.get("motivation", {}) or {}
        motivation_meta = motivation_block.get("metadata", {}) or {}
        motivation_state = motivation_block.get("state", {}) or {}

        # Barriers block
        barriers_block = meta.get("barriers", {}) or {}
        barriers_meta = barriers_block.get("metadata", {}) or {}
        barriers_state = barriers_block.get("state", {}) or {}

        # Intent fields
        intent_status = intent_meta.get("confirm_intent") or intent_meta.get("intent_status", "—")
        intent_type = intent_meta.get("intent_type", "—")

        # Tone fields
        confirm_tone = tone_meta.get("confirm_tone") or "unclear"
        tone_type = tone_meta.get("emo_tone") or tone_meta.get("emo_tone_type") or "—"
        emo_tone_1 = tone_state.get("emo_tone_1") or "—"
        emo_tone_2 = tone_state.get("emo_tone_2") or "—"
        emo_tone_3 = tone_state.get("emo_tone_3") or "—"

        # Motivation fields
        confirm_motivation = motivation_meta.get("confirm_motivation") or "unclear"
        motivation_type = motivation_meta.get("motivation_type", "—")
        motivation_1 = motivation_state.get("motivation_1") or "—"
        motivation_2 = motivation_state.get("motivation_2") or "—"
        motivation_3 = motivation_state.get("motivation_3") or "—"
        motivation_4 = motivation_state.get("motivation_4") or "—"

        # Barriers fields
        confirm_barriers = barriers_meta.get("confirm_barriers") or "unclear"
        barrier_type = barriers_meta.get("barrier_type", "—")
        barrier_1 = barriers_state.get("barrier_1") or "—"
        barrier_2 = barriers_state.get("barrier_2") or "—"
        barrier_3 = barriers_state.get("barrier_3") or "—"
        barrier_4 = barriers_state.get("barrier_4") or "—"

        # Display insights
        st.markdown(f"- **Confirm Intent:** {intent_status or '—'}")
        st.markdown(f"- **Intent Type:** {intent_type or '—'}")

        st.markdown(f"- **Confirm Tone:** {confirm_tone}")
        st.markdown(f"- **Tone Type:** {tone_type}")

        st.markdown(f"- **Confirm Motivation:** {confirm_motivation}")
        st.markdown(f"- **Motivation Type:** {motivation_type}")

        st.markdown(f"- **Confirm Barriers:** {confirm_barriers}")
        st.markdown(f"- **Barrier Type:** {barrier_type}")
        # Summary block
        summary_block = meta.get("summary", {}) or {}
        summary_meta = summary_block.get("metadata", {}) or {}

        # Summary fields
        confirm_summary = summary_meta.get("confirm_summary") or "unclear"

        st.markdown(f"- **Confirm Summary:** {confirm_summary}")
        # Optional: show separate debug states
        st.markdown("---")
        st.markdown("**Debug States:**")

        # INTENT STATE - Remove intent_mapping and options
        intent_state_clean = {k: v for k, v in intent_state.items() if k not in ['intent_mapping', 'options']}
        st.markdown(f"- **INTENT STATE:** {json.dumps(intent_state_clean, ensure_ascii=False)}")

        # TONE STATE - Remove emotional_mapping and options
        tone_state_clean = {k: v for k, v in tone_state.items() if k not in ['emotional_mapping', 'options']}
        st.markdown(f"- **TONE STATE:** {json.dumps(tone_state_clean, ensure_ascii=False)}")

        # MOTIVATION STATE - Remove motivation_mapping and options
        motivation_state_clean = {k: v for k, v in motivation_state.items() if
                                  k not in ['motivation_mapping', 'options']}
        st.markdown(f"- **MOTIVATION STATE:** {json.dumps(motivation_state_clean, ensure_ascii=False)}")

        # Show individual motivation answers clearly
        if any([motivation_1 != "—", motivation_2 != "—", motivation_3 != "—", motivation_4 != "—"]):
            st.markdown("**Motivation Answers:**")
            if motivation_1 != "—":
                st.markdown(f"  - Motivation 1: {motivation_1}")
            if motivation_2 != "—":
                st.markdown(f"  - Motivation 2: {motivation_2}")
            if motivation_3 != "—":
                st.markdown(f"  - Motivation 3: {motivation_3}")
            if motivation_4 != "—":
                st.markdown(f"  - Motivation 4: {motivation_4}")

        # BARRIERS STATE - Remove barriers_mapping and options
        barriers_state_clean = {k: v for k, v in barriers_state.items() if k not in ['barriers_mapping', 'options']}
        st.markdown(f"- **BARRIERS STATE:** {json.dumps(barriers_state_clean, ensure_ascii=False)}")

        # Show individual barrier answers clearly
        if any([barrier_1 != "—", barrier_2 != "—", barrier_3 != "—", barrier_4 != "—"]):
            st.markdown("**Barrier Answers:**")
            if barrier_1 != "—":
                st.markdown(f"  - Barrier 1: {barrier_1}")
            if barrier_2 != "—":
                st.markdown(f"  - Barrier 2: {barrier_2}")
            if barrier_3 != "—":
                st.markdown(f"  - Barrier 3: {barrier_3}")
            if barrier_4 != "—":
                st.markdown(f"  - Barrier 4: {barrier_4}")

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
            st.markdown(t["content"])

    # ========== CHECK IF CONVERSATION IS COMPLETE ==========
    stage_meta = st.session_state.state.get("stage_meta", {}) or {}

    # Check intent status
    intent_block = stage_meta.get("connection_intent", {}) or {}
    intent_meta = intent_block.get("metadata", {}) or {}
    confirm_intent = intent_meta.get("confirm_intent", "unclear")
    intent_type = intent_meta.get("intent_type", "")

    # Check emotional tone status - try emotional_tone first, fallback to connection_tone
    tone_block = stage_meta.get("emotional_tone", {}) or stage_meta.get("connection_tone", {}) or {}
    tone_meta = tone_block.get("metadata", {}) or {}
    confirm_tone = tone_meta.get("confirm_tone", "unclear")
    emo_tone = tone_meta.get("emo_tone", "")

    # Check motivation status
    motivation_block = stage_meta.get("motivation", {}) or {}
    motivation_meta = motivation_block.get("metadata", {}) or {}
    confirm_motivation = motivation_meta.get("confirm_motivation", "unclear")
    motivation_type = motivation_meta.get("motivation_type", "")

    # Check barriers status
    barriers_block = stage_meta.get("barriers", {}) or {}
    barriers_meta = barriers_block.get("metadata", {}) or {}
    confirm_barriers = barriers_meta.get("confirm_barriers", "unclear")
    barrier_type = barriers_meta.get("barrier_type", "")

    # Conversation is complete when all four are clear
    # Check summary status
    summary_block = stage_meta.get("summary", {}) or {}
    summary_meta = summary_block.get("metadata", {}) or {}
    confirm_summary = summary_meta.get("confirm_summary", "unclear")
    summary_text = summary_meta.get("summary_text", "")
    recommendations = summary_meta.get("recommendations", [])

    # Conversation is complete when all FIVE are clear (including summary)
    conversation_ended = (confirm_intent == "clear" and confirm_tone == "clear" and
                          confirm_motivation == "clear" and confirm_barriers == "clear" and
                          confirm_summary == "clear")

    # ========== SHOW BUTTONS OR COMPLETION MESSAGE ==========
    if conversation_ended:
        # Show completion message
        st.success(f"✅ **Conversation Complete!**")
        st.info(
            f"**Intent:** {intent_type} | **Emotional Tone:** {emo_tone} | **Motivation:** {motivation_type} | **Barrier:** {barrier_type}")

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
        # Buttons for current options (only if conversation is ongoing)
        options = st.session_state.state.get("last_options", []) or []
        if options and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write("Choose an option:")
                cols = st.columns(len(options))
                for i, opt in enumerate(options):
                    if cols[i].button(opt, key=f"opt_{i}"):
                        st.session_state.chat_history.append({"role": "user", "content": opt})
                        ai_text, new_state = process_user_message(
                            st.session_state.graph, st.session_state.state, opt
                        )
                        st.session_state.state = new_state

                        # Check if conversation just ended
                        new_stage_meta = new_state.get("stage_meta", {}) or {}
                        new_intent_meta = new_stage_meta.get("connection_intent", {}).get("metadata", {})
                        new_tone_meta = new_stage_meta.get("emotional_tone", {}).get("metadata",
                                                                                     {}) or new_stage_meta.get(
                            "connection_tone", {}).get("metadata", {})
                        new_motivation_meta = new_stage_meta.get("motivation", {}).get("metadata", {})
                        new_barriers_meta = new_stage_meta.get("barriers", {}).get("metadata", {})

                        new_confirm_intent = new_intent_meta.get("confirm_intent", "unclear")
                        new_confirm_tone = new_tone_meta.get("confirm_tone", "unclear")
                        new_confirm_motivation = new_motivation_meta.get("confirm_motivation", "unclear")
                        new_confirm_barriers = new_barriers_meta.get("confirm_barriers", "unclear")

                        # Check if conversation is complete (all 4 agents done)
                        if (new_confirm_intent == "clear" and new_confirm_tone == "clear" and
                                new_confirm_motivation == "clear" and new_confirm_barriers == "clear"):
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
                st.markdown(ai_text)
            st.rerun()

if __name__ == "__main__":
    main()