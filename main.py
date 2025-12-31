# app_enhanced_refactored.py – Sparky with inheritance-based agent architecture
from __future__ import annotations
from collections import Counter

import os, json, re, yaml, glob
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Any, Tuple, Sequence, Annotated, Literal
from abc import ABC, abstractmethod

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import random
from langchain_core.pydantic_v1 import validator

# from langchain_core.pydantic_v1 import BaseModel as PydanticV1BaseModel, Field as PydanticV1Field


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


# ============================================================================
# CACHING FUNCTIONS - Solution 1: Cache YAML Config
# ============================================================================

@st.cache_data
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Cache YAML config to avoid re-reading file on every Streamlit rerun.
    This only reads the file ONCE, then reuses the cached version.

    Savings: 50-200ms per rerun
    """
    config_path_obj = Path(config_path)
    if config_path_obj.exists():
        with open(config_path_obj, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        print(f"WARNING: Config file not found at {config_path}. Using empty config.")
        return {}


# Load agent + supervisor configuration from YAML (CACHED)
CONFIG_PATH = Path(DEFAULT_CONFIG_PATH)
_raw_cfg = load_yaml_config(str(CONFIG_PATH))  # ✅ Now uses cached version
AGENT_CONFIGS: Dict[str, Dict[str, Any]] = _raw_cfg.get("agents", {})
SUPERVISOR_PROMPTS: Dict[str, str] = _raw_cfg.get("supervisor", {})


# ============================================================================
# CACHING FUNCTIONS - Solution 1 & 2: Cache LLM Client and Graph
# ============================================================================

@st.cache_resource
def get_cached_llm():
    """
    Cache the LLM client to avoid recreating connection on every rerun.
    Uses @st.cache_resource so the same ChatOpenAI instance is reused.

    Savings: ~500ms per agent/supervisor call
    """
    if not USE_OPENAI or ChatOpenAI is None:
        return None
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=2500,
    )


@st.cache_resource
def get_cached_graph():
    """
    Cache the entire LangGraph structure to avoid rebuilding on every rerun.
    Uses @st.cache_resource so the same graph is reused.

    Savings: ~800ms on initialization
    """
    return create_graph(AGENT_CONFIGS)


@st.cache_resource
def preload_everything():
    """Pre-load all resources into cache on startup"""
    import time
    start = time.time()

    print("\n" + "=" * 60)
    print("🔄 PRE-LOADING RESOURCES INTO CACHE...")
    print("=" * 60)

    # Load YAML
    step_start = time.time()
    config = load_yaml_config(str(CONFIG_PATH))
    print(f"✅ YAML config loaded: {(time.time() - step_start) * 1000:.0f}ms")

    # Create LLM
    step_start = time.time()
    llm = get_cached_llm()
    print(f"✅ LLM client created: {(time.time() - step_start) * 1000:.0f}ms")

    total_time = (time.time() - start) * 1000
    print(f"✅ PRE-LOADING COMPLETE: {total_time:.0f}ms")
    print("=" * 60 + "\n")

    return {"config": config, "llm": llm, "loaded_at": time.time()}


# Call it immediately
PRELOADED = preload_everything()


# ============================================================================
# CACHING FUNCTIONS - Solution 3: Cache LLM Responses Per Session
# ============================================================================

def get_cached_llm_response(agent_name: str, turn: int, llm, messages):
    """
    Cache LLM responses in session state per user.

    Args:
        agent_name: Name of the agent (e.g., 'act_1', 'act_2', 'supervisor')
        turn: Current turn/question number for this agent
        llm: The LLM instance
        messages: Messages to send to LLM

    Returns:
        The LLM response (cached or fresh)

    Savings: 1-3 seconds per cached question + API cost savings
    """
    # Initialize cache if not exists
    if 'llm_cache' not in st.session_state:
        st.session_state.llm_cache = {}

    # Create unique cache key for this agent + turn
    cache_key = f"{agent_name}_turn_{turn}"

    # Check if we have a cached response
    if cache_key in st.session_state.llm_cache:
        print(f"✅ Cache HIT: {cache_key} - Using cached response")
        return st.session_state.llm_cache[cache_key]

    # Not cached - make API call
    print(f"🔄 Cache MISS: {cache_key} - Calling OpenAI API...")
    response = llm.invoke(messages)

    # Store in cache for this user's session
    st.session_state.llm_cache[cache_key] = response

    return response


def clear_llm_cache():
    """
    Clear the LLM response cache for the current user session.
    Use this when starting a new conversation to ensure fresh responses.
    """
    if 'llm_cache' in st.session_state:
        st.session_state.llm_cache = {}
        print("🗑️ LLM cache cleared")


def reset_conversation():
    """
    Reset the entire conversation and clear all caches.
    This gives the user a fresh start.
    """
    # Clear LLM response cache
    clear_llm_cache()

    # Reset conversation state
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
        conversation_history=[],
        user_age="",
        user_gender="",
        selected_courses=[],
        course_responses={},
        course_selection_complete=False,
    )

    base_state = ensure_act_2_block(base_state)
    st.session_state.state = base_state
    st.session_state.chat_history = []
    st.session_state.conversation_complete = False

    print("🔄 Conversation reset complete")


def get_image_path_from_metadata(state: AgentState) -> str:
    """
    Get the appropriate image path based on current agent and mapping.
    Falls back to random image from folder if mapping is incorrect.
    """
    import random

    stage_meta = state.get("stage_meta", {}) or {}
    last_agent = state.get("last_agent", "")
    print(f"DEBUG: last_agent = {last_agent}")

    # Default fallback
    default_image = "welcome.webp"

    # Map agent names to Act folders
    agent_to_act = {
        "act_1": "Act1",
        "act_2": "Act2",
        "act_3": "Act3",
        "act_4": "Act4"
    }

    # Check if current agent is one we handle
    if last_agent not in agent_to_act:
        print(f"DEBUG: Agent not in map, returning default")
        return default_image

    act_folder = agent_to_act[last_agent]
    print(f"DEBUG: act_folder = {act_folder}")

    # Get the agent's state block
    agent_block = stage_meta.get(last_agent, {}) or {}
    agent_state = agent_block.get("state", {}) or {}

    # Determine question number based on turn
    turn = agent_state.get("turn", 0)
    print(f"DEBUG: turn = {turn}")

    # Map turns to questions for each agent
    if last_agent == "act_1":
        if turn in [1, 2]:
            question_folder = "Q1"  # Aspiration questions
        elif turn in [3, 4]:
            question_folder = "Q3"  # Identity questions
        else:
            question_folder = "Q1"  # Default

    elif last_agent == "act_2":
        if turn == 1:
            question_folder = "Q1"
        elif turn == 2:
            question_folder = "Q2"
        elif turn in [3, 4]:
            question_folder = "Q3"
        elif turn == 5:
            question_folder = "Q5"
        elif turn >= 6:
            question_folder = "Q6"
        else:
            question_folder = "Q1"


    elif last_agent == "act_3":
        if turn in [1, 2]:
            question_folder = "Q1"  # Broad Internal Fear
        elif turn in [3, 4]:
            question_folder = "Q3"  # Broad Emotional Pattern
        else:
            question_folder = "Q1"
    elif last_agent == "act_4":
        if turn == 1:
            question_folder = "Q1"  # Broad Support
        elif turn == 2:
            question_folder = "Q2"  # Focused Support
        else:
            question_folder = "Q1"
    else:
        question_folder = "Q1"

    print(f"DEBUG: question_folder = {question_folder}")

    # Get the mapping from current question's options (not previous answer)
    mapping = None

    # Get the option_mapping for the CURRENT question being displayed
    option_mapping = agent_state.get("option_mapping", [])

    print(f"DEBUG: option_mapping for current question = {option_mapping}")

    # If we have mappings, randomly choose one to display
    # (since we don't know which the user will pick yet)
    if option_mapping and len(option_mapping) > 0:
        # Filter out empty strings
        valid_mappings = [m for m in option_mapping if m and m.strip()]
        if valid_mappings:
            import random
            mapping = random.choice(valid_mappings)
            print(f"DEBUG: Randomly selected mapping from current options: {mapping}")

    print(f"DEBUG: final mapping = {mapping}")

    # Construct the image path
    folder_path = os.path.join(act_folder, question_folder)
    print(f"DEBUG: folder_path = {folder_path}")


    # Try to use the mapping first
    if mapping and os.path.exists(folder_path):
        image_path = os.path.join(folder_path, f"{mapping}.png")
        print(f"DEBUG: trying mapped image: {image_path}")
        if os.path.exists(image_path):
            print(f"DEBUG: Found mapped image! Returning: {image_path}")
            return image_path

    # Fallback: randomly choose from available images in folder
    if os.path.exists(folder_path):
        available_images = glob.glob(os.path.join(folder_path, "*.png"))
        print(f"DEBUG: available_images = {available_images}")
        if available_images:
            selected = random.choice(available_images)
            print(f"DEBUG: Randomly selected: {selected}")
            return selected

    # Final fallback
    print(f"DEBUG: No image found, returning default: {default_image}")
    return default_image


def format_affirmation(affirmation: str) -> str:
    """Format affirmation text in red, bold, bigger, and centered for Streamlit markdown."""
    if affirmation:
        return f"<div style='text-align: center; margin-bottom: 40px;'><strong><span style='color:red; font-size:36px;'>{affirmation}</span></strong></div>"
    return ""


def format_hook_text(hook_text: str) -> str:
    """Format hook text with larger font size for Streamlit markdown."""
    if hook_text:
        return f"<span style='font-size:30px'>{hook_text}</span>"
    return ""


def format_question_text(question_text: str) -> str:
    """Format question text with larger font size and center alignment."""
    if question_text:
        return f"<div style='text-align: center; font-size: 36px;'>{question_text}</div>"
    return ""


def format_conversation_history(conversation_history: List[Dict[str, str]]) -> str:
    """Format the last 4 turns of conversation history for prompt injection."""
    print(f"DEBUG CONV HISTORY - Raw: {conversation_history}")  # ADD THIS
    if not conversation_history:
        return "No previous conversation."

    formatted = []
    for i, turn in enumerate(conversation_history[-4:], 1):  # Get last 4 turns
        # Skip if turn is not a dictionary
        if not isinstance(turn, dict):
            continue

        affirmation = turn.get('affirmation', '')
        question = turn.get('question', '')
        answer = turn.get('answer', '')

        formatted.append(f"Turn {i}:")
        if affirmation:
            formatted.append(f"  Affirmation: {affirmation}")
        if question:
            formatted.append(f"  Question: {question}")
        if answer:
            formatted.append(f"  User Answer: {answer}")

    if not formatted:
        return "No previous conversation."

    return "\n".join(formatted)


# ============================================================================
# IMAGE SELECTION HELPER FUNCTIONS
# ============================================================================

def select_facet_from_existing_bank(ad_theme: str, q1_identity_cluster: str, q2_scale_1_to_5: int) -> str:
    """
    Returns a facet that EXISTS in the current image bank folders.
    Uses: identity -> aspiration mapping, presence from Q2, and deterministic fallbacks.

    Args:
        ad_theme: Theme from ad data (currently not used, kept for future)
        q1_identity_cluster: The identity cluster selected in Q1
        q2_scale_1_to_5: Scale value from Q2 (1-5)

    Returns:
        Facet name that exists in the image bank
    """
    # 1) Identity -> Aspiration (align these keys to your exact identity cluster names)
    identity_to_aspiration = {
        "finishes_what_they_start": "confidence_progress",
        "real_creative_hobby": "enrichment_purpose",
        "seen_as_creative": "self_expression",
        "invests_in_self": "enrichment_purpose",
        "learns_new_skills": "exploration",
        "time_for_self": "calm_wellbeing",
        "real_confidence": "confidence_progress",
        "creates_with_hands": "self_expression",
        "open_to_trying": "exploration",
        "more_interesting_self": "enrichment_purpose",
        "shows_world_creations": "self_expression",
        "pushes_comfort_zone": "exploration",
        "unlock_creative_potential": "inspiration",
        "just_knows_how_to_do_things": "confidence_progress"
    }

    aspiration = identity_to_aspiration.get(q1_identity_cluster, "enrichment_purpose")

    # 2) Q2 -> presence bucket
    if q2_scale_1_to_5 in (1, 2):
        presence = "low"
    elif q2_scale_1_to_5 == 3:
        presence = "medium"
    else:
        presence = "high"

    # 3) Available facets (matches your current folders)
    available = {
        "inspiration": {"low": "dormant_potential", "medium": "creative_spark"},
        "confidence_progress": {"medium": "incremental_improvement"},
        "calm_wellbeing": {"high": "low_stimulation"},
        "exploration": {"low": "open_experiment", "medium": "variety_play"},
        "self_expression": {"high": "active_creation"},
        "enrichment_purpose": {"all": "life_expansion"}
    }

    # 4) Resolve facet with fallbacks
    asp_map = available.get(aspiration)
    if not asp_map:
        # ultimate fallback if aspiration missing entirely
        return "life_expansion"

    # If aspiration has an "all" facet, always use it
    if "all" in asp_map:
        return asp_map["all"]

    # Exact match
    if presence in asp_map:
        return asp_map[presence]

    # Closest-presence fallback
    if presence == "low":
        fallback_order = ["medium", "high"]
    elif presence == "medium":
        fallback_order = ["low", "high"]
    else:  # high
        fallback_order = ["medium", "low"]

    for p in fallback_order:
        if p in asp_map:
            return asp_map[p]

    # If somehow none matched
    return next(iter(asp_map.values()))


def get_q3_images(aspiration_category: str, facet: str) -> List[Tuple[str, str]]:
    import os
    import glob

    # This ensures it finds the folder inside your current project directory
    base_dir = os.getcwd()
    base_path = os.path.join(base_dir, "Image questions", "Act 1 Q3")

    # Build path: base_path / aspiration / facet
    image_folder = os.path.join(base_path, aspiration_category, facet)

    if not os.path.exists(image_folder):
        print(f"⚠️ WARNING: Image folder not found: {image_folder}")
        return []

    image_files = glob.glob(os.path.join(image_folder, "*.png"))

    if not os.path.exists(image_folder):
        print(f"⚠️ WARNING: Image folder not found: {image_folder}")
        return []

    # Get all PNG files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))

    if not image_files:
        print(f"⚠️ WARNING: No images found in: {image_folder}")
        return []

    # Return list of (full_path, mapping_name) tuples
    results = []
    for img_path in image_files:
        # Extract filename without extension (this is the mapping)
        filename = os.path.basename(img_path)
        mapping_name = os.path.splitext(filename)[0]  # e.g., "grounding"
        results.append((img_path, mapping_name))

    print(f"✅ Found {len(results)} images in {aspiration_category}/{facet}")
    return results

# ============================================================================
# COURSE SELECTION CONSTANTS
# ============================================================================


def select_learning_screens(
        act4_support_theme: str = None,
        act1_aspiration: str = None,
        act2_learning_style: str = None,
        act3_pattern: str | None = None,
        max_screens: int = 5
) -> List[str]:
    """
    Select course screens to show user.
    Currently uses random selection - designed to be easily replaceable
    with agent-based or sophisticated logic later.

    Args:
        act4_support_theme: Support theme from act 4 (currently not used)
        act1_aspiration: Aspiration from act 1 (currently not used)
        act2_learning_style: Learning style from act 2 (currently not used)
        act3_pattern: Pattern from act 3 (currently not used)
        max_screens: Number of screens to return (default: 5)

    Returns:
        List of randomly selected course names in random order
    """

    # All available courses (matching folder names in "Image questions/Courses/")
    all_courses = [
        "Candy",
        "Chocolate",
        "Face_cream",
        "Herbal_remedies",
        "Organic Soap",
        "Painting",
        "Sew_cloths"
    ]

    # Random selection - REPLACE THIS BLOCK when implementing smart selection
    selected = random.sample(all_courses, min(max_screens, len(all_courses)))

    return selected

COURSE_QUESTIONS = {
    "Candy": "Would You Like to Learn How to Make Candy in Just 20 Minutes?",
    "Chocolate": "Would You Like to Learn How to Make Your Own Chocolate in Just 20 Minutes?",
    "Face_cream": "Would You Like to Learn How to Make Your Own Natural Face Cream in Just 20 Minutes?",
    "Herbal_remedies": "Would You Like to Learn How to Make Your Own Herbal Remedies in Just 20 Minutes?",
    "Organic Soap": "Would You Like to Know How to Make Organic Soap in Just 20 Minutes?",
    "Painting": "Would You Like to Learn How to Create a Beautiful Painting in Just 20 Minutes?",
    "Sew_cloths": "Would You Like to Learn How to Sew Your Own Clothes in Just 20 Minutes?"
}
COURSE_IMAGES = {
    "Candy": "Image questions/Courses/Candy.png",
    "Chocolate": "Image questions/Courses/Chocolate.png",
    "Face_cream": "Image questions/Courses/Face_cream.png",
    "Herbal_remedies": "Image questions/Courses/Herbal_remedies.png",
    "Organic Soap": "Image questions/Courses/Organic Soap.png",
    "Painting": "Image questions/Courses/Painting.png",
    "Sew_cloths": "Image questions/Courses/Sew_cloths.png"
}


def select_courses_for_user(state: AgentState) -> List[str]:
    """
    Wrapper function to select courses for the user.

    This is the SINGLE POINT OF REPLACEMENT when you want to:
    - Add an LLM agent to make smart selections
    - Implement sophisticated scoring logic
    - Use any other selection strategy

    Currently just calls select_learning_screens() with random selection.

    Args:
        state: Current AgentState with all user data from previous acts

    Returns:
        List of 5 course names to show the user
    """
    # Extract data from state if needed for future smart selection
    # act_4_data = state.get("stage_meta", {}).get("act_4", {})
    # act_1_data = state.get("stage_meta", {}).get("act_1", {})
    # etc.

    # For now, just use random selection
    # REPLACE THIS FUNCTION BODY when implementing smart selection
    selected_courses = select_learning_screens()

    return selected_courses
# ============================================================================
# AD DATA INTEGRATION
# ============================================================================


AdTheme = Literal["self_expression", "wellness", "skill_growth", "ambition", "belonging"]


class AdData(TypedDict):
    ad_name: str
    ad_description: str
    ad_theme: AdTheme
    ad_tone: str

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
        "ad_tone": "awe, mastery, achievement",
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
        "ad_tone": "relief, empowerment, validation"
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
        "ad_tone": "reflective, searching, meaningful",
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
        "ad_tone": "reflective, searching, meaningful",
    },
    "narrator_and_lily": {
        "ad_name": "Heartfelt Connection",
        "ad_description": (
            "A warm, family-oriented storytelling ad centered on connection, presence, and shared creative "
            "moments. It appeals to parents, sentimental adults, and nostalgic creatives who value bonding "
            "and meaningful experiences. Creativity is framed as a way to nurture relationships and build memories."
        ),
        "ad_theme": "enrichment_purpose",
        "ad_tone": "warm, connecting, present",
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
        "ad_tone": "empowering, practical, opportunity-driven",
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

    # Act 2 answer tracking fields (Q6-Q11)
    act_2_state.setdefault("act_2_answer_1", "")  # Q6 - Entry Style
    act_2_state.setdefault("act_2_answer_2", "")  # Q7 - Momentum Pattern
    act_2_state.setdefault("act_2_answer_3", "")  # Q8 - Situational Disruption
    act_2_state.setdefault("act_2_answer_4", "")  # Q9 - Time & Energy Fit
    act_2_state.setdefault("act_2_answer_5", "")  # Q10 - Learning Mode Recognition
    act_2_state.setdefault("act_2_answer_6", "")  # Q11 - Confidence & Safety Signal

    # Derived learning behavior signals (internal only, never shown to user)
    act_2_state.setdefault("entry_style", "")  # From Q6
    act_2_state.setdefault("momentum_support", "")  # From Q7
    act_2_state.setdefault("situational_friction", "")  # From Q8
    act_2_state.setdefault("time_energy_fit", "")  # From Q9
    act_2_state.setdefault("learning_mode", "")  # From Q10
    act_2_state.setdefault("emotional_safety_level", "")  # From Q11

    # Response format tracking for each question
    act_2_state.setdefault("response_format_1", "multiple_choice")
    act_2_state.setdefault("response_format_2", "multiple_choice")
    act_2_state.setdefault("response_format_3", "multiple_choice")
    act_2_state.setdefault("response_format_4", "scale")  # Q9 is typically scale
    act_2_state.setdefault("response_format_5", "image_select")  # Q10 can be image
    act_2_state.setdefault("response_format_6", "scale")  # Q11 is typically scale
    act_2_state.setdefault("scale_range", "")  # e.g., "1-5" when using scales
    act_2_state.setdefault("scale_labels", {})  # e.g., {"min": "...", "max": "..."}

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
# ============================================================================
# PYDANTIC MODELS - ACT 1 RESPONSE
# ============================================================================


from pydantic import validator


class Act1Response(PydanticV1BaseModel):
    """
    Response format for Act 1 agent (Q1-Q5).
    Supports multiple response formats based on question number.
    """
    affirmations: List[str] = PydanticV1Field(
        description="Array of neutral, validating affirmations (18-24 words each), one for each answer option"
    )
    affirmation: Optional[str] = PydanticV1Field(
        default=None,
        description="DEPRECATED: Backwards compatibility only"
    )
    question_text: str = PydanticV1Field(description="Question text")
    response_format: Literal["multiple_choice", "scale", "image_select", "either_or"] = PydanticV1Field(
        description="Question format type"
    )

    # For multiple_choice (Q1) and either_or (Q5)
    options: Optional[List[str]] = PydanticV1Field(
        default=None,
        description="REQUIRED for multiple_choice/either_or: Answer options (4 items for Q1, 2 items for Q5)"
    )
    option_mapping: str = PydanticV1Field(
        default="",
        description="REQUIRED for Q1 multiple_choice: Comma-separated mappings in format 'identity|aspiration,identity|aspiration,...'. Example: 'finishes_what_they_start|confidence_progress,learns_new_skills|exploration,real_creative_hobby|enrichment_purpose,more_interesting_self|enrichment_purpose'. For Q5: 'aligned,misaligned'"
    )



    # For scale (Q2, Q4)
    scale_range: str = PydanticV1Field(
        default="",
        description="REQUIRED for scale format: MUST be '1-5'. Never omit this field for scale questions."
    )


    # For image_select (Q3) - NO options, NO mappings
    # Images are selected by Python code upstream, not by LLM
class AgentResponse(PydanticV1BaseModel):
    """Generic structured response used by ALL agents.

    - affirmations: array of warm reflections/validations (one per option).
    - affirmation: DEPRECATED - backwards compatibility only, use affirmations instead.
    - question_text: conversational question or prompt.
    - options: list of answer choices.
    - option_mapping: category mappings for each option (agent-specific categories).
    - metadata: agent-specific output (intent, act_3_type, barriers, persona, etc.).
    - state: agent-specific state (turn counts, levels, flags, progress).
    """

    affirmations: Optional[List[str]] = PydanticV1Field(
        default=None,
        description="Array of affirmations, one for each answer option (typically 4 items)"
    )

    # BACKWARDS COMPATIBILITY: Accept old single affirmation format
    affirmation: Optional[str] = PydanticV1Field(
        default=None,
        description="DEPRECATED: Single affirmation (use affirmations array instead)"
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
        min_items=0,
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

    def get_affirmations(self) -> List[str]:
        """
        Get affirmations array with backwards compatibility.
        If affirmations is provided, use it.
        If only old affirmation is provided, convert it to array.
        """
        if self.affirmations:
            return self.affirmations
        elif self.affirmation:
            # Convert single affirmation to array (duplicate for all options)
            return [self.affirmation] * 4
        else:
            return []

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

    response_format: str = PydanticV1Field(
        default="multiple_choice",
        description="Type of response: 'multiple_choice', 'either_or', 'scale', 'image_choice'"
    )

    # Make options optional for scale questions
    options: Optional[List[str]] = PydanticV1Field(
        default=None,
        description="Answer options (4 for multiple_choice/image_choice, 2 for either_or, None for scale)"
    )

    option_mapping: Optional[List[str]] = PydanticV1Field(
        default=None,
        description=(
            "Intent categories matching options. "
            "Values: 'inspiration', 'exploration', 'confidence_progress', 'calm_wellbeing', "
            "'routine_structure', 'self_expression', 'enrichment_purpose', "
            "'creative_identity', 'confidence_identity', 'growth_identity', "
            "'hands_on_identity', 'openness_identity', 'visibility_identity', "
            "'capability_identity', 'discipline_identity', 'lifestyle_identity', 'unsure'. "
            "Required for all formats except scale."
        )
    )

    # Scale-specific fields
    scale_range: Optional[str] = PydanticV1Field(
        default=None,
        description="Scale range like '1-5' or '1-10' (only for scale format)"
    )

    scale_labels: Optional[Dict[str, str]] = PydanticV1Field(
        default=None,
        description="Optional labels for scale endpoints, e.g., {'min': 'Not at all', 'max': 'Very much'}"
    )

    scale_mapping: Optional[Dict[str, str]] = PydanticV1Field(
        default=None,
        description="Mapping of number ranges to intent categories. Keys are ranges like '1-3', '4-7', values are categories. Only for scale format."
    )

    # Image-specific fields
    image_urls: Optional[List[str]] = PydanticV1Field(
        default=None,
        description="Image URLs for image_choice format (4 images matching 4 options)"
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
        description="Mapping of number ranges to emotional categories. Keys are ranges like '1-3', '4-7', values are categories like 'neutral', 'tense'. Only for scale format."
    )

    class Config:
        extra = "forbid"


# Hook Response for hook agent
class HookResponse(PydanticV1BaseModel):
    hook_text: str = PydanticV1Field(description="The 2-3 sentence hook message")

    class Config:
        extra = "forbid"


class MotivationResponse(PydanticV1BaseModel):
    """Strict response format for motivation agent (act_3).
    Supports multiple response formats: multiple_choice, scale, and either_or."""

    affirmation: str = PydanticV1Field(
        ...,  # Required field (no default)
        description="Warm reflection/validation sentence (1-2 sentences)",
        min_length=10
    )

    question_text: str = PydanticV1Field(
        ...,  # Required field
        description="First-person question about internal fears or emotional patterns",
        min_length=10
    )

    response_format: str = PydanticV1Field(
        default="multiple_choice",
        description="Type of response expected: 'multiple_choice', 'scale', or 'either_or'"
    )

    # Fields for multiple_choice and either_or formats
    options: Optional[List[str]] = PydanticV1Field(
        default=None,
        description="Answer options (4 for multiple_choice, 2 for either_or, None for scale)"
    )

    act_3_mapping: Optional[List[str]] = PydanticV1Field(
        default=None,
        description=(
            "Fear or pattern clusters matching the options. "
            "Values must be one of: "
            "'fear_not_good_enough', 'fear_failure', 'fear_starting', 'fear_not_finishing', 'fear_judgment', "
            "'perfectionism_cycle', 'shame_inconsistency', 'low_creative_esteem', 'disconnection_potential', 'emotional_freeze'. "
            "Required for multiple_choice and either_or, None for scale."
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
        description="Mapping of number ranges to fear/pattern intensity categories. Only for scale format."
    )

    class Config:
        extra = "forbid"

# Barriers Response for barriers agent
class BarriersResponse(PydanticV1BaseModel):
    """Strict response format for barriers agent - supports multiple_choice, scale, and either_or."""

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

    response_format: str = PydanticV1Field(
        ...,  # Required field
        description="Must be one of: 'multiple_choice', 'scale', 'either_or'"
    )

    # For multiple_choice and either_or
    options: Optional[List[str]] = PydanticV1Field(
        default=None,
        description="Answer options (4 for multiple_choice, 2 for either_or, empty for scale)"
    )

    act_4_mapping: Optional[List[str]] = PydanticV1Field(
        default=None,
        description=(
            "Barrier categories matching options. "
            "Must match the support clusters from the YAML: 'feeling_capable', 'easy_start', "
            "'proud_progress', 'gentle_pacing', 'staying_connected', 'inspired_flow', 'making_room'. "
            "Example: ['easy_start', 'proud_progress', 'staying_connected', 'inspired_flow']"
        )
    )

    # For scale questions
    scale_range: Optional[str] = PydanticV1Field(
        default=None,
        description="Scale range (e.g., '1-5') - only for scale questions"
    )

    scale_labels: Optional[Dict[str, str]] = PydanticV1Field(
        default=None,
        description="Min/max labels - only for scale questions. Example: {'min': 'not important', 'max': 'very important'}"
    )

    scale_mapping: Optional[Dict[str, str]] = PydanticV1Field(
        default=None,
        description="Map scale ranges to categories - only for scale questions. Example: {'1-2': 'low', '3': 'medium', '4-5': 'high'}"
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


def select_q10_image_set(
        entry_style: str,
        momentum_support: str,
        time_energy_fit: str,
        ad_tone: str
) -> Dict[str, str]:
    """
    Determine which image set to show for Act 2 Q10 based on derived learning behavior signals.

    Args:
        entry_style: How user naturally begins (from Q6)
        momentum_support: What helps user continue (from Q7)
        time_energy_fit: How this fits into daily life (from Q9)
        ad_tone: Tone from ad data

    Returns:
        Dictionary with:
            - image_category: learning mode category (hands_on, watch_first, hybrid, exploratory)
            - image_variant: visual style (compact, immersive)
            - image_set_id: folder name format "{learning_mode}__{image_variant}"
            - tone: ad_tone passed through
    """
    # Step 1: Determine learning_mode from entry_style and momentum_support
    if entry_style == "observe_first":
        learning_mode = "watch_first"
    elif entry_style == "action_first" and momentum_support in ["progress_driven", "frequency_driven"]:
        learning_mode = "hands_on"
    elif entry_style == "mixed_entry":
        learning_mode = "hybrid"
    elif momentum_support == "flexibility_driven":
        learning_mode = "exploratory"
    else:
        learning_mode = "hybrid"  # safe default

    # Step 2: Determine image_variant from time_energy_fit
    if time_energy_fit in ["tight_fit", "variable_fit"]:
        image_variant = "compact"
    else:
        image_variant = "immersive"

    # Step 3: Build image_set_id (this matches folder structure)
    image_set_id = f"{learning_mode}__{image_variant}"

    return {
        "image_category": learning_mode,
        "image_variant": image_variant,
        "image_set_id": image_set_id,
        "tone": ad_tone
    }


def update_act_1_metadata_after_answer(state: AgentState, user_answer: str) -> AgentState:
    """
    Update intent metadata after user answers Q1-Q5, BEFORE the agent runs again.

    This function handles:
    - Q1: Extract identity & aspiration from option_mapping
    - Q2: Map scale value to presence_level
    - Q3: Store selected image mapping and facet
    - Q4: Map scale value to context_influence
    - Q5: Store alignment status, set confirm_act_1 = "clear"
    """
    import copy

    stage_meta = copy.deepcopy(state.get("stage_meta", {}) or {})
    act_1_block = stage_meta.get("act_1", {}) or {}
    act_1_state = dict(act_1_block.get("state", {}) or {})
    act_1_meta = dict(act_1_block.get("metadata", {}) or {})

    current_turn = act_1_state.get("turn", 0)

    if not user_answer:
        return state

    ad_data = state.get("ad_data", {}) or {}
    ad_theme = ad_data.get("ad_theme", "")

    print(f"🔍 ACT_1 UPDATE - Turn {current_turn}, Answer: '{user_answer}'")

    # Inside update_act_1_metadata_after_answer in main.py
    if current_turn == 1:
        # Get the nested mapping we asked the LLM to generate
        nested_mapping = act_1_meta.get("nested_option_mapping", [])
        options = act_1_state.get("options", [])
        selected_index = -1
        user_lower = user_answer.lower().strip()

        # 1. Find which button was clicked
        for i, opt in enumerate(options):
            if opt and (opt.lower() in user_lower or user_lower in opt.lower()):
                selected_index = i
                break

        # 2. Extract folder slugs (with hardcoded fallback if LLM mapping is empty)
        if 0 <= selected_index < len(nested_mapping) and nested_mapping[selected_index]:
            mapping_dict = nested_mapping[selected_index]
            identity = mapping_dict.get("identity_shift_cluster", "")
            aspiration = mapping_dict.get("aspiration_category", "")
        else:
            # FALLBACK: If LLM sent an empty list, map the text manually to ensure Q3 works
            fallback_map = {
                "finishes what they start": ("finishes_what_they_start", "confidence_progress"),
                "learns new skills": ("learns_new_skills", "exploration"),
                "creative hobby": ("real_creative_hobby", "enrichment_purpose"),
                "interesting version": ("more_interesting_self", "enrichment_purpose")
            }
            identity, aspiration = "unsure", "enrichment_purpose"
            for key, values in fallback_map.items():
                if key in user_lower:
                    identity, aspiration = values
                    break

        # 3. Store in state so get_q3_images can find the right folder
        act_1_state["selected_identity_cluster"] = identity
        act_1_state["mapped_aspiration_category"] = aspiration
        print(f"✅ Q1 FIXED: Identity={identity}, Folder={aspiration}")

    # Q2: Scale - Map to presence_level
    elif current_turn == 2:
        if user_answer.isdigit():
            scale_value = int(user_answer)
            scale_mapping = act_1_state.get("scale_mapping", {})

            presence_level = ""
            for range_str, category in scale_mapping.items():
                if "-" in range_str:
                    min_val, max_val = map(int, range_str.split("-"))
                    if min_val <= scale_value <= max_val:
                        presence_level = category
                        break

            act_1_state["presence_level"] = presence_level
            act_1_state["q2_answer"] = scale_value
            act_1_state["q2_presence"] = presence_level

            print(f"✅ Q2: scale={scale_value}, presence={presence_level}")

    # Q3: Image select - Store selected mapping
    elif current_turn == 3:
        # user_answer should be the image path
        # Extract mapping from filename
        if "/" in user_answer or "\\" in user_answer:
            filename = os.path.basename(user_answer)
            mapping_name = os.path.splitext(filename)[0]

            act_1_state["q3_answer"] = user_answer
            act_1_state["q3_mapping"] = mapping_name

            # Also store the facet that was used
            identity_cluster = act_1_state.get("selected_identity_cluster", "")
            presence_level = act_1_state.get("presence_level", "")
            q2_answer = act_1_state.get("q2_answer", 3)

            # Reconstruct facet
            facet = select_facet_from_existing_bank(ad_theme, identity_cluster, q2_answer)
            act_1_state["q3_facet"] = facet

            print(f"✅ Q3: image={mapping_name}, facet={facet}")

    # Q4: Scale - Map to context_influence
    elif current_turn == 4:
        if user_answer.isdigit():
            scale_value = int(user_answer)
            scale_mapping = act_1_state.get("scale_mapping", {})

            context_influence = ""
            for range_str, category in scale_mapping.items():
                if "-" in range_str:
                    min_val, max_val = map(int, range_str.split("-"))
                    if min_val <= scale_value <= max_val:
                        context_influence = category
                        break

            act_1_state["context_influence"] = context_influence
            act_1_state["q4_answer"] = scale_value
            act_1_state["q4_context"] = context_influence

            print(f"✅ Q4: scale={scale_value}, context={context_influence}")

    # Q5: Either-or - Store alignment status
    elif current_turn == 5:
        option_mapping = act_1_state.get("option_mapping", [])
        options = act_1_state.get("options", [])

        # Find which option user selected
        selected_index = -1
        user_lower = user_answer.lower().strip()

        if len(user_lower) == 1 and user_lower in ['a', 'b']:
            selected_index = ord(user_lower) - ord('a')
        else:
            for i, opt in enumerate(options):
                if opt and (opt.lower() in user_lower or user_lower in opt.lower()):
                    selected_index = i
                    break

        # Extract alignment status
        if 0 <= selected_index < len(option_mapping):
            mapping_dict = option_mapping[selected_index]
            if isinstance(mapping_dict, dict):
                alignment_status = mapping_dict.get("alignment", "")
            else:
                # Fallback if mapping is just a string
                alignment_status = "aligned" if selected_index == 0 else "misaligned"

            act_1_state["alignment_status"] = alignment_status
            act_1_state["q5_answer"] = user_answer
            act_1_state["q5_alignment"] = alignment_status

            # Q5 is the last question - finalize Act 1
            act_1_meta["confirm_act_1"] = "clear"

            # Compute final act_1_type based on all answers
            # Use aspiration from Q1 as primary, fallback to ad_theme
            aspiration = act_1_state.get("mapped_aspiration_category", ad_theme)
            act_1_meta["act_1_type"] = aspiration if aspiration else "unsure"

            print(f"✅ Q5: alignment={alignment_status}, FINALIZED act_1_type={act_1_meta['act_1_type']}")

    # Write back updated state
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
    # Determine which format was used for this question (Q6-Q11)
    if current_turn == 1:
        response_format = act_2_state.get("response_format_1", "multiple_choice")
    elif current_turn == 2:
        response_format = act_2_state.get("response_format_2", "multiple_choice")
    elif current_turn == 3:
        response_format = act_2_state.get("response_format_3", "multiple_choice")
    elif current_turn == 4:
        response_format = act_2_state.get("response_format_4", "scale")
    elif current_turn == 5:
        response_format = act_2_state.get("response_format_5", "image_select")
    elif current_turn == 6:
        response_format = act_2_state.get("response_format_6", "scale")
    else:
        response_format = "multiple_choice"

    # Classify the user's answer and derive learning behavior signal
    derived_signal = ""

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
                        derived_signal = category
                        break
                elif range_str.isdigit() and int(range_str) == scale_value:
                    derived_signal = category
                    break
    elif response_format == "image_select":
        # For Q10 image selection, derive learning_mode from image path
        # The learning_mode will be derived using select_q10_image_set logic
        # Store the raw answer (image path) for now
        derived_signal = user_answer  # Will be processed later
    else:
        # For multiple_choice and either_or, use option_mapping
        prev_option_mapping = act_2_state.get("option_mapping", [])
        prev_options = act_2_state.get("options", [])

        # Filter out empty strings
        prev_option_mapping = [m for m in prev_option_mapping if m]
        prev_options = [o for o in prev_options if o]

        if prev_option_mapping and prev_options:
            # Find which option the user selected
            user_lower = user_answer.lower().strip()
            for i, option in enumerate(prev_options):
                option_lower = option.lower().strip()
                if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
                    if i < len(prev_option_mapping):
                        derived_signal = prev_option_mapping[i]
                        break

    print(f"DEBUG - update_act_2_metadata: turn={current_turn}, answer='{user_answer}', signal='{derived_signal}'")

    # Store the answer and derived signal for each question (Q6-Q11)
    if current_turn == 1:
        # Q6 - Entry Style
        act_2_state["act_2_answer_1"] = user_answer
        act_2_state["entry_style"] = derived_signal
        act_2_state["last_theme"] = derived_signal
    elif current_turn == 2:
        # Q7 - Momentum Support
        act_2_state["act_2_answer_2"] = user_answer
        act_2_state["momentum_support"] = derived_signal
        act_2_state["last_theme"] = derived_signal
    elif current_turn == 3:
        # Q8 - Situational Friction
        act_2_state["act_2_answer_3"] = user_answer
        act_2_state["situational_friction"] = derived_signal
        act_2_state["last_theme"] = derived_signal
    elif current_turn == 4:
        # Q9 - Time & Energy Fit
        act_2_state["act_2_answer_4"] = user_answer
        act_2_state["time_energy_fit"] = derived_signal
        act_2_state["last_theme"] = derived_signal
    elif current_turn == 5:
        # Q10 - Learning Mode (from image or choice)
        # ===== FIX: For image_select, extract semantic mapping value =====
        if response_format == "image_select":
            # Extract semantic value from mapping
            prev_option_mapping = act_2_state.get("act_2_emotional_mapping", [])
            prev_options = act_2_state.get("options", [])

            if user_answer in prev_options and len(prev_option_mapping) == len(prev_options):
                idx = prev_options.index(user_answer)
                semantic_value = prev_option_mapping[idx]
                act_2_state["act_2_answer_5"] = semantic_value  # Store semantic value
                act_2_state["learning_mode"] = semantic_value
                act_2_state["last_theme"] = semantic_value
                print(f"DEBUG - Act 2 Turn 5: Image → semantic value '{semantic_value}'")
            else:
                # Fallback: store the answer as-is
                act_2_state["act_2_answer_5"] = user_answer
                act_2_state["learning_mode"] = derived_signal
                act_2_state["last_theme"] = derived_signal
        else:
            act_2_state["act_2_answer_5"] = user_answer
            act_2_state["learning_mode"] = derived_signal
            act_2_state["last_theme"] = derived_signal

    elif current_turn == 6:
        # Q11 - Emotional Safety Level
        act_2_state["act_2_answer_6"] = user_answer
        act_2_state["emotional_safety_level"] = derived_signal
        act_2_state["last_theme"] = derived_signal

    # Check if we should finalize (after 6 questions)
    if current_turn >= 6:
        # We're done with Act 2 questions
        act_2_meta["confirm_act_2"] = "clear"
        # No single "final tone" - we have 6 derived signals instead
    else:
        # Keep asking questions
        act_2_meta["confirm_act_2"] = "unclear"
    # Write back the updated metadata and state
    stage_meta["act_2"] = {
        "metadata": act_2_meta,
        "state": act_2_state,
    }

    return {
        **state,
        "stage_meta": stage_meta,
    }


def get_user_chosen_act_3(
        user_answer: str,
        options: List[str],
        act_3_mapping: List[str]
) -> str:
    """
    Match user's answer to the motivation mapping.
    Returns the motivation category, or 'unsure' if not found.
    """
    # ===== FIX: Handle empty options (scale questions handled elsewhere) =====
    if not options or len(options) == 0:
        print(f"WARNING - get_user_chosen_act_3: empty options, returning 'unsure'")
        return "unsure"

    # ===== FIX: Validate act_3_mapping length =====
    if act_3_mapping and len(act_3_mapping) not in [2, 4]:
        print(f"WARNING - get_user_chosen_act_3: invalid act_3_mapping: {act_3_mapping}")
        return "unsure"

    if not act_3_mapping or len(act_3_mapping) == 0:
        print(f"WARNING - get_user_chosen_act_3: empty act_3_mapping")
        return "unsure"

    # Try to find exact match
    if user_answer in options:
        idx = options.index(user_answer)
        if idx < len(act_3_mapping):
            return act_3_mapping[idx]

    # Try fuzzy matching as fallback
    user_lower = user_answer.lower().strip()
    for i, opt in enumerate(options):
        if user_lower in opt.lower() or opt.lower() in user_lower:
            if i < len(act_3_mapping):
                return act_3_mapping[i]

    print(f"WARNING - get_user_chosen_act_3: couldn't match '{user_answer}' to options")
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


def should_finalize_act_3(act_3_answer_1: str, act_3_answer_2: str, act_3_answer_3: str, act_3_answer_4: str,
                          act_1_type: str) -> Tuple[bool, str]:
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
    final_motivation = compute_final_motivation(act_3_answer_1, act_3_answer_2, act_3_answer_3, act_3_answer_4,
                                                act_1_type)

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

    # Only process if the user actually answered
    if not user_answer:
        return state

    # ===== FIX: Handle scale questions differently =====
    response_format = act_3_state.get("response_format", "")

    if response_format == "scale":
        # For scale questions, convert numeric answer to semantic value
        try:
            numeric_value = int(user_answer)
        except (TypeError, ValueError):
            numeric_value = None
            chosen_act_3 = "unsure"

        if numeric_value is not None:
            scale_mapping = act_3_state.get("scale_mapping", {})
            chosen_act_3 = str(numeric_value)  # fallback

            # Map numeric value to semantic value
            for key, value in scale_mapping.items():
                if "-" in key:
                    try:
                        lo, hi = map(int, key.split("-"))
                        if lo <= numeric_value <= hi:
                            chosen_act_3 = value
                            break
                    except ValueError:
                        continue
                elif str(numeric_value) == key:
                    chosen_act_3 = value
                    break

            print(f"DEBUG - Act 3 Scale: numeric={numeric_value} → semantic='{chosen_act_3}'")
    else:
        # For non-scale questions, classify using mapping
        if not prev_act_3_mapping:
            return state

        chosen_act_3 = get_user_chosen_act_3(user_answer, prev_options, prev_act_3_mapping)

    print(
        f"DEBUG - update_act_3_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_act_3='{chosen_act_3}'")

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
        options: List of option texts (4 for multiple_choice, 2 for either_or, empty for scale)
        act_4_mapping: List of barrier categories corresponding to options

    Returns:
        The barrier category the user selected
    """
    # ===== FIX: Handle empty options (scale questions handled elsewhere) =====
    if not options or len(options) == 0:
        print(f"WARNING - get_user_chosen_act_4: empty options, returning 'unsure'")
        return "unsure"

    # ===== FIX: Validate act_4_mapping length for multiple_choice (4) or either_or (2) =====
    if act_4_mapping and len(act_4_mapping) not in [2, 4]:
        print(f"WARNING - get_user_chosen_act_4: invalid act_4_mapping: {act_4_mapping}")
        return "unsure"

    if not act_4_mapping or len(act_4_mapping) == 0:
        print(f"WARNING - get_user_chosen_act_4: empty act_4_mapping")
        return "unsure"

    # Validate options and mapping have same length
    if len(options) != len(act_4_mapping):
        print(f"WARNING - get_user_chosen_act_4: options and mapping length mismatch")
        return "unsure"

    user_lower = user_input.lower().strip()

    # Try to match user input to one of the options
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        # Exact match or significant overlap
        if user_lower == option_lower or user_lower in option_lower or option_lower in user_lower:
            if i < len(act_4_mapping):
                return act_4_mapping[i]

    # Fallback: couldn't determine
    print(f"WARNING - get_user_chosen_act_4: couldn't match '{user_input}' to options")
    return "unsure"

def compute_question_direction_act_4(act_4_answer_1: str,
                                     current_turn: int) -> str:
    """
    Determine if next barrier question should be 'broad' or 'focused'.

    Logic (4 questions):
    - Turn 1: Always broad (exploring support needs)
    - Turn 2: Focused if act_4_answer_1 is clear and not unsure, broad otherwise
    - Turn 3: Always broad (exploring support style)
    - Turn 4: Focused if act_4_answer_3 is clear and not unsure, broad otherwise

    Args:
        act_4_answer_1: User's support preference from Q1
        current_turn: The question number we're about to ask (1-4)

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

    if current_turn == 3:
        return "broad"  # Start new exploration of support style

    if current_turn == 4:
        # Turn 4 is always focused
        return "focused"

    # Default to broad as fallback
    return "broad"

def compute_final_barrier(act_4_answer_1: str, act_4_answer_2: str, act_4_answer_3: str, act_4_answer_4: str,
                          act_3_type: str) -> str:
    """
    Determine the final barrier from 4 answers.

    Logic:
    1. If 3+ answers are "unsure" → "unsure"
    2. If same barrier appears 3+ times → that barrier
    3. If tie (2 barriers with 2 votes each) → use most recent
    4. If all different → use most recent

    Args:
        act_4_answer_1: Barrier from Q1
        act_4_answer_2: Barrier from Q2
        act_4_answer_3: Barrier from Q3
        act_4_answer_4: Barrier from Q4
        act_3_type: User's motivation (used as tiebreaker if needed)

    Returns:
        Final barrier category
    """
    # Normalize inputs
    barriers = [
        (act_4_answer_1 or "").lower().strip(),
        (act_4_answer_2 or "").lower().strip(),
        (act_4_answer_3 or "").lower().strip(),
        (act_4_answer_4 or "").lower().strip()
    ]

    # If any are missing, return unsure
    if not all(barriers):
        return "unsure"

    # RULE 1: If 3+ answers are "unsure" → "unsure"
    unsure_count = sum(1 for b in barriers if b == "unsure")
    if unsure_count >= 3:
        return "unsure"

    # RULE 2: Count occurrences
    from collections import Counter

    # Filter out empty and unsure
    filtered = [b for b in barriers if b and b != "unsure"]

    if not filtered:
        return "unsure"

    counts = Counter(filtered)
    max_count = max(counts.values())

    # If we have a clear winner (appears 3+ times)
    if max_count >= 3:
        candidates = [b for b, c in counts.items() if c == max_count]
        return candidates[0]

    # RULE 3: Tie or no clear winner - use most recent
    # Priority order: act_4_answer_4 > act_4_answer_3 > act_4_answer_2 > act_4_answer_1
    priority_order = [barriers[3], barriers[2], barriers[1], barriers[0]]
    for p in priority_order:
        if p in filtered:
            return p

    # Fallback
    return barriers[3] if barriers[3] else "unsure"


def match_affirmation_to_answer(user_response, options, affirmations):
    # Scale question: map scale value to appropriate affirmation
    if not options or len(options) == 0:
        if affirmations and len(affirmations) == 4:
            try:
                # Extract scale value from "2 (presence level: low_presence)"
                scale_value = int(user_response.split()[0])

                # Map scale 1-5 to affirmation indices 0-3
                if scale_value <= 2:
                    return affirmations[0]  # LOW
                elif scale_value == 3:
                    return affirmations[1]  # MEDIUM-LOW
                elif scale_value == 4:
                    return affirmations[2]  # MEDIUM-HIGH
                else:  # scale_value == 5
                    return affirmations[3]  # HIGH
            except:
                return affirmations[1] if len(affirmations) > 1 else affirmations[0]
        return "Thank you for sharing that."

    # Multiple choice: existing logic
    if not affirmations or len(affirmations) != len(options):
        return "Thank you for sharing that."

    try:
        index = options.index(user_response)
        return affirmations[index]
    except (ValueError, IndexError):
        return "Thank you for sharing that."

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
    if not act_4_answer_4 or act_4_answer_4 == "":
        return (False, "")

    # We have 4 answers, time to finalize
    # Call the logic function to determine final barrier
    final_barrier = compute_final_barrier(act_4_answer_1, act_4_answer_2, act_4_answer_3, act_4_answer_4, act_3_type)

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
    course_selection_complete = st.session_state.state.get("course_selection_complete", False)  # ADD THIS LINE

    # Get stored mapping from previous agent run
    prev_act_4_mapping = act_4_state.get("act_4_mapping", [])
    prev_options = act_4_state.get("options", [])

    # Get current turn (which question did they just answer?)
    current_turn = act_4_state.get("turn", 0)

    # Only process if the user actually answered
    if not user_answer:
        return state

    # ===== FIX: Handle scale questions differently =====
    response_format = act_4_state.get("response_format", "")

    if response_format == "scale":
        # For scale questions, convert numeric answer to semantic value
        try:
            numeric_value = int(user_answer)
        except (TypeError, ValueError):
            numeric_value = None
            chosen_act_4 = "unsure"

        if numeric_value is not None:
            scale_mapping = act_4_state.get("scale_mapping", {})
            chosen_act_4 = str(numeric_value)  # fallback

            # Map numeric value to semantic value
            for key, value in scale_mapping.items():
                if "-" in key:
                    try:
                        lo, hi = map(int, key.split("-"))
                        if lo <= numeric_value <= hi:
                            chosen_act_4 = value
                            break
                    except ValueError:
                        continue
                elif str(numeric_value) == key:
                    chosen_act_4 = value
                    break

            print(f"DEBUG - Act 4 Scale: numeric={numeric_value} → semantic='{chosen_act_4}'")
    else:
        # For non-scale questions, classify using mapping
        if not prev_act_4_mapping:
            return state

        chosen_act_4 = get_user_chosen_act_4(user_answer, prev_options, prev_act_4_mapping)

    print(
        f"DEBUG - update_act_4_metadata_after_answer: turn={current_turn}, user_answer='{user_answer}', chosen_act_4='{chosen_act_4}'")

    # Update support preference based on which question they just answered
    if current_turn == 1:
        act_4_state["act_4_answer_1"] = chosen_act_4
        act_4_state["last_act_4"] = chosen_act_4
        act_4_state["last_theme"] = chosen_act_4
    elif current_turn == 2:
        act_4_state["act_4_answer_2"] = chosen_act_4
        act_4_state["last_act_4"] = chosen_act_4
        act_4_state["last_theme"] = chosen_act_4
    elif current_turn == 3:
        act_4_state["act_4_answer_3"] = chosen_act_4
        act_4_state["last_act_4"] = chosen_act_4
        act_4_state["last_theme"] = chosen_act_4
    elif current_turn == 4:
        act_4_state["act_4_answer_4"] = chosen_act_4
        act_4_state["last_act_4"] = chosen_act_4
        act_4_state["last_theme"] = chosen_act_4

    # Get all four support preferences
    act_4_answer_1 = act_4_state.get("act_4_answer_1", "")
    act_4_answer_2 = act_4_state.get("act_4_answer_2", "")
    act_4_answer_3 = act_4_state.get("act_4_answer_3", "")
    act_4_answer_4 = act_4_state.get("act_4_answer_4", "")

    # Get act_3_type for finalization logic
    act_3_block = stage_meta.get("act_3", {}) or {}
    act_3_meta = act_3_block.get("metadata", {}) or {}
    act_3_type = act_3_meta.get("act_3_type", "")

    # Check if we should finalize (after 4 questions)
    should_finalize, final_barrier = should_finalize_act_4(
        act_4_answer_1, act_4_answer_2, act_4_answer_3, act_4_answer_4, act_3_type
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
    conversation_history: List[Dict[str, str]]  # Track last 4 Q&A turns

    # Course selection state (between act_4 and summary)
    selected_courses: List[str]  # List of 5 course names selected
    course_responses: Dict[str, str]  # Map of course_name -> "Yes" or "No"
    course_selection_complete: bool  # True when all 5 courses have been shown

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
    try:
        result = (tmpl or "").format_map(_SafeDict(**ctx))
        return result
    except ValueError as e:
        print(f"❌ ERROR in render_tmpl:")
        print(f"   Error: {e}")
        print(f"   Template length: {len(tmpl or '')}")
        print(f"   Template preview: {(tmpl or '')[:500]}")
        print(f"   Context keys: {list(ctx.keys())}")
        for key, val in ctx.items():
            val_str = str(val)[:100]
            print(f"   ctx['{key}']: {repr(val_str)}")
        raise





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

        # Format conversation history for prompt injection
        conversation_history = state.get("conversation_history", [])
        formatted_history = format_conversation_history(conversation_history)

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
            "conversation_history": formatted_history,
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

            # Get current turn to determine question number
            current_turn = act_1_state.get("turn", 0)
            ad_data = state.get("ad_data", {}) or {}
            ctx["ad_theme"] = ad_data.get("ad_theme", "")
            ctx["ad_tone"] = ad_data.get("ad_tone", "")
            # Add Act 1 specific context for Q1-Q5
            ctx["question_number"] = current_turn + 1  # Next question (1-5)
            ctx["user_last_answer"] = user_text
            ctx["selected_identity_cluster"] = act_1_state.get("selected_identity_cluster", "")
            ctx["mapped_aspiration_category"] = act_1_state.get("mapped_aspiration_category", "")
            ctx["presence_level"] = act_1_state.get("presence_level", "")

            # Determine question_type based on question_number
            question_type_map = {
                1: "multiple_choice",
                2: "scale",
                3: "image_select",
                4: "scale",
                5: "either_or"
            }
            ctx["question_type"] = question_type_map.get(ctx["question_number"], "multiple_choice")

            print(f"🔍 ACT_1 Context Built - Q{ctx['question_number']} ({ctx['question_type']})")
            print(f"   Identity: {ctx['selected_identity_cluster']}")
            print(f"   Aspiration: {ctx['mapped_aspiration_category']}")
            print(f"   Presence: {ctx['presence_level']}")

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

            # Add context for new Act 2 questions (Q6-Q11)
            ctx["question_number"] = current_turn + 1  # Next question (1-6)
            ctx["user_last_answer"] = user_text

            # Get Act 2 derived signals so far
            ctx["mapped_aspiration_category"] = act_1_state.get("mapped_aspiration_category", "")
            ctx["presence_level"] = act_1_state.get("presence_level", "")

            # Get already-derived Act 2 signals for context
            act2_signal_state = {
                "entry_style": act_2_state.get("entry_style", ""),
                "momentum_support": act_2_state.get("momentum_support", ""),
                "situational_friction": act_2_state.get("situational_friction", ""),
                "time_energy_fit": act_2_state.get("time_energy_fit", ""),
                "learning_mode": act_2_state.get("learning_mode", ""),
                "emotional_safety_level": act_2_state.get("emotional_safety_level", "")
            }
            ctx["act2_signal_state"] = json.dumps(act2_signal_state)

            # Determine question_mode (broad/focused) - simplified for now
            # Act 2 doesn't use the same broad/focused pattern as Act 1
            ctx["question_mode"] = "standard"

            # For Q10 (turn 5), prepare image selection context
            if current_turn + 1 == 5:
                # Q10 - Learning Mode Recognition (image_select)
                # Use helper function to determine which images to show
                entry_style = act_2_state.get("entry_style", "")
                momentum_support = act_2_state.get("momentum_support", "")
                time_energy_fit = act_2_state.get("time_energy_fit", "")

                ad_data = state.get("ad_data", {}) or {}
                ad_tone = ad_data.get("ad_tone", "neutral")

                # Call helper function to get image set
                from pathlib import Path
                image_info = select_q10_image_set(
                    entry_style=entry_style,
                    momentum_support=momentum_support,
                    time_energy_fit=time_energy_fit,
                    ad_tone=ad_tone
                )

                # Build image paths based on image_set_id
                image_set_id = image_info["image_set_id"]
                image_directory = Path("Act 2 Q10") / image_set_id

                # Get 4 images from this directory
                if image_directory.exists():
                    image_files = list(image_directory.glob("*.png")) + list(image_directory.glob("*.jpg"))
                    image_urls = [str(f) for f in image_files[:4]]  # Take first 4 images
                else:
                    # Fallback if directory doesn't exist
                    image_urls = []

                ctx["image_urls"] = image_urls
                ctx["learning_mode_category"] = image_info["image_category"]

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

            # ADD THESE DEBUG LINES:
            print(f"🔍 DEBUG ACT_3 CONTEXT:")
            print(f"   last_theme: {repr(last_act_3)}")
            print(f"   question_mode: {repr(question_mode)}")
            print(f"   focus_type: {repr(focus_type)}")
            print(f"   act1_identity: {repr(act1_identity)}")
            print(f"   conversation_history type: {type(ctx.get('conversation_history'))}")

            # SHOW FULL CONVERSATION HISTORY:
            conv_hist = ctx.get('conversation_history', '')
            print(f"   conversation_history FULL LENGTH: {len(conv_hist)}")
            print(f"   conversation_history FULL CONTENT:")
            print(f"   {repr(conv_hist)}")

            # CHECK FOR CURLY BRACES:
            if '{' in conv_hist or '}' in conv_hist:
                print(f"   ⚠️ WARNING: Found curly braces in conversation_history!")
                print(f"   Positions of {{: {[i for i, c in enumerate(conv_hist) if c == '{']}")
                print(f"   Positions of }}: {[i for i, c in enumerate(conv_hist) if c == '}']}")
            # Get Act 2 derived psychographic fields for act_3
            act_2_block = stage_meta.get("act_2", {}) or {}
            act_2_state = act_2_block.get("state", {}) or {}

            act2_learning_style = act_2_state.get("act2_learning_style", "")
            act2_engagement_style = act_2_state.get("act2_engagement_style", "")
            act2_final_tone = act_2_state.get("act2_final_tone", "")

            ctx["act2_learning_style"] = str(act2_learning_style)
            ctx["act2_engagement_style"] = str(act2_engagement_style)
            ctx["act2_final_tone"] = str(act2_final_tone)
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
            act_4_answer_4 = act_4_state.get("act_4_answer_4", "")
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
            # All four turns focus on "support"
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

            # Add Act 2 learning behavior data to context (6 answers + 6 signals)
            ctx["act_2_answer_1"] = act_2_state.get("act_2_answer_1", "")
            ctx["act_2_answer_2"] = act_2_state.get("act_2_answer_2", "")
            ctx["act_2_answer_3"] = act_2_state.get("act_2_answer_3", "")
            ctx["act_2_answer_4"] = act_2_state.get("act_2_answer_4", "")
            ctx["act_2_answer_5"] = act_2_state.get("act_2_answer_5", "")
            ctx["act_2_answer_6"] = act_2_state.get("act_2_answer_6", "")

            # Derived learning behavior signals
            ctx["entry_style"] = act_2_state.get("entry_style", "")
            ctx["momentum_support"] = act_2_state.get("momentum_support", "")
            ctx["situational_friction"] = act_2_state.get("situational_friction", "")
            ctx["time_energy_fit"] = act_2_state.get("time_energy_fit", "")
            ctx["learning_mode"] = act_2_state.get("learning_mode", "")
            ctx["emotional_safety_level"] = act_2_state.get("emotional_safety_level", "")

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
            ctx["act_4_answer_3"] = act_4_state.get("act_4_answer_3", "")
            ctx["act_4_answer_4"] = act_4_state.get("act_4_answer_4", "")

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
            # Use cached LLM instead of creating new instance
            llm = get_cached_llm()
            if llm is None:
                raise Exception("LLM not available")

            # Get current turn for cache key
            stage_meta = state.get("stage_meta", {}) or {}
            my_block = stage_meta.get(self.name, {}) or {}
            my_state = my_block.get("state", {}) or {}
            current_turn = my_state.get("turn", 0)

            # Use strict schema for hook agent
            if self.info_type == "hook":
                structured_llm = llm.with_structured_output(HookResponse)
                # Cache the response
                raw_response = get_cached_llm_response(self.name, current_turn, structured_llm, msgs)
                strict_response: HookResponse = raw_response
                return AgentResponse(
                    affirmation="",
                    question_text=strict_response.hook_text,
                    options=[],
                    option_mapping=["", "", "", ""],
                    metadata={"hook_text": strict_response.hook_text},
                    state={},
                )

            # CHANGE 5: CONNECTION_INTENT AGENT (Act 1 Q1-Q5)
            elif self.info_type == "connection_intent":
                # Use Act1Response for Q1-Q5
                structured_llm = llm.with_structured_output(Act1Response)
                raw_response = get_cached_llm_response(self.name, current_turn, structured_llm, msgs)
                strict_response: Act1Response = raw_response

                print(f"✅ ACT_1 Response - Format: {strict_response.response_format}")

                # Build AgentResponse based on format
                response_format = strict_response.response_format

                # Store format in metadata
                act_1_metadata = {
                    "response_format": response_format
                }

                # Q1: Multiple choice with nested identity→aspiration mappings
                if response_format == "multiple_choice" and current_turn + 1 == 1:
                    # Extract nested mappings from string
                    options = list(strict_response.options or [])
                    option_mapping_flat = []
                    nested_mapping_for_storage = []

                    # Parse the option_mapping string
                    # Format: "identity1|aspiration1,identity2|aspiration2,identity3|aspiration3,identity4|aspiration4"
                    if strict_response.option_mapping and strict_response.option_mapping.strip():
                        mappings = strict_response.option_mapping.split(",")
                        for mapping_str in mappings:
                            mapping_str = mapping_str.strip()
                            if "|" in mapping_str:
                                parts = mapping_str.split("|")
                                identity = parts[0].strip()
                                aspiration = parts[1].strip() if len(parts) > 1 else ""

                                # Add to flat list (just identity)
                                option_mapping_flat.append(identity)

                                # Store as dict for nested mapping
                                nested_mapping_for_storage.append({
                                    "identity_shift_cluster": identity,
                                    "aspiration_category": aspiration
                                })
                            else:
                                # Fallback if no pipe separator
                                option_mapping_flat.append("")
                                nested_mapping_for_storage.append({})

                    # Pad to 4
                    while len(option_mapping_flat) < 4:
                        option_mapping_flat.append("")
                    while len(options) < 4:
                        options.append("")
                    while len(nested_mapping_for_storage) < 4:
                        nested_mapping_for_storage.append({})

                    # Store the parsed nested mappings in metadata
                    act_1_metadata["nested_option_mapping"] = nested_mapping_for_storage

                    response = AgentResponse(
                        affirmations=strict_response.affirmations,
                        question_text=strict_response.question_text,
                        options=options,
                        option_mapping=option_mapping_flat,
                        metadata=act_1_metadata,
                        state={}
                    )

                # Q2 & Q4: Scale questions
                elif response_format == "scale":
                    # Hardcode scale metadata (these never change)
                    SCALE_LABELS = {"min": "Rarely", "max": "Very often"}
                    SCALE_MAPPING = {"1-2": "low_presence", "3": "medium_presence", "4-5": "high_presence"}

                    act_1_metadata = {
                        "response_format": "scale",
                        "scale_range": strict_response.scale_range or "1-5",
                        "scale_labels": SCALE_LABELS,
                        "scale_mapping": SCALE_MAPPING
                    }

                    response = AgentResponse(
                        affirmations=strict_response.affirmations,
                        question_text=strict_response.question_text,
                        options=[],
                        option_mapping=["", "", "", ""],
                        metadata=act_1_metadata,
                        state={}
                    )
                    # --- Update inside BaseAgent.generate_response ---

                    # Q3: Image select - Python selects images, injecting them into options
                elif response_format == "image_select":
                    # 1. Access the current state to know what to look for
                    act_1_state = stage_meta.get("act_1", {}).get("state", {})
                    identity_cluster = act_1_state.get("selected_identity_cluster", "")
                    aspiration = act_1_state.get("mapped_aspiration_category", "")
                    q2_answer = act_1_state.get("q2_answer", 3)
                    ad_data = state.get("ad_data", {}) or {}
                    ad_theme = ad_data.get("ad_theme", "")

                    # 2. Use your helpers to find the files on disk
                    facet = select_facet_from_existing_bank(ad_theme, identity_cluster, q2_answer)
                    image_list = get_q3_images(aspiration, facet)

                    # 3. Extract the paths and mapping names
                    image_paths = [img[0] for img in image_list]
                    image_mappings = [img[1] for img in image_list]

                    # 4. CRITICAL: Inject the paths into 'options' for the UI to render
                    return AgentResponse(
                        affirmations=strict_response.affirmations,
                        question_text=strict_response.question_text,
                        options=image_paths[:4],  # These MUST be the file paths for st.image
                        option_mapping=image_mappings[:4],  # These are the tags for your logic
                        metadata={
                            "response_format": "image_select",
                            "selected_facet": facet
                        },
                        state={}
                    )




                # Q5: Either-or with alignment mapping
                elif response_format == "either_or":
                    options = list(strict_response.options or [])
                    # For Q5, option_mapping should be ["aligned", "misaligned"]
                    option_mapping_flat = []
                    if strict_response.option_mapping:
                        for mapping_dict in strict_response.option_mapping:
                            if isinstance(mapping_dict, dict):
                                alignment = mapping_dict.get("alignment", "")
                                option_mapping_flat.append(alignment)
                            elif isinstance(mapping_dict, str):
                                option_mapping_flat.append(mapping_dict)
                            else:
                                option_mapping_flat.append("")

                    # Pad to 4
                    while len(option_mapping_flat) < 4:
                        option_mapping_flat.append("")
                    while len(options) < 4:
                        options.append("")

                    # Store original nested mappings in metadata
                    act_1_metadata["nested_option_mapping"] = strict_response.option_mapping

                    response = AgentResponse(
                        affirmations=strict_response.affirmations,
                        question_text=strict_response.question_text,
                        options=options,
                        option_mapping=option_mapping_flat,
                        metadata=act_1_metadata,
                        state={}
                    )

                else:
                    # Fallback for any other format
                    options = list(strict_response.options or [])
                    option_mapping = list(strict_response.option_mapping or [])

                    while len(option_mapping) < 4:
                        option_mapping.append("")
                    while len(options) < 4:
                        options.append("")

                    response = AgentResponse(
                        affirmation=strict_response.affirmations,
                        question_text=strict_response.question_text,
                        options=options,
                        option_mapping=option_mapping,
                        metadata=act_1_metadata,
                        state={}
                    )
                return response
            # Use strict schema for emotional_tone agent
            elif self.info_type == "emotional_tone":
                structured_llm = llm.with_structured_output(EmotionalToneResponse)
                # Cache the response
                raw_response = get_cached_llm_response(self.name, current_turn, structured_llm, msgs)
                strict_response: EmotionalToneResponse = raw_response

                # Convert to AgentResponse format
                # Store response_format and scale-specific data in metadata
                act_2_metadata = {
                    "response_format": strict_response.response_format
                }
                # Get Act 2 state to access previous answers
                act_2_state = stage_meta.get("act_2", {}).get("state", {})


                # Add scale-specific fields if this is a scale question
                if strict_response.response_format == "scale":
                    act_2_metadata["scale_range"] = strict_response.scale_range
                    act_2_metadata["scale_labels"] = strict_response.scale_labels or {}
                    act_2_metadata["scale_mapping"] = strict_response.scale_mapping or {}

                act_2_emotional_mapping = list(strict_response.act_2_emotional_mapping or [])
                while len(act_2_emotional_mapping) < 4:
                    act_2_emotional_mapping.append("")  # Pad with empty strings

                # Pad options to match if needed
                options = list(strict_response.options or [])
                while len(options) < 4:
                    options.append("")  # Pad with empty strings

                print(f"🔍 DEBUG BEFORE Q10 CHECK: current_turn={current_turn}, type={type(current_turn)}")
                print(f"🔍 DEBUG: strict_response.response_format={strict_response.response_format}")

                # SPECIAL HANDLING FOR Q10 (Turn 5) - FORCE IMAGE_SELECT WITH THEME-BASED SUBFOLDER
                if current_turn == 4:  # Turn 4 = Q10 (5th question of Act 2)
                    print(f"🎯 INSIDE Q10 BLOCK! Loading images...")
                    # Force image_select format even if LLM gave wrong format
                    act_2_metadata["response_format"] = "image_select"

                    # Load images from Act 2 Q10 folder with theme-based subfolder selection
                    base_dir = os.getcwd()
                    q10_base_folder = os.path.join(base_dir, "Image questions", "Act 2 Q10")

                    # Determine subfolder based on previous answers
                    learning_style = None
                    intensity = None

                    # Map Q6 answer (act_2_answer_1) to learning style
                    q6_answer = act_2_state.get('act_2_answer_1', '')
                    if 'jump' in q6_answer.lower() or 'right in' in q6_answer.lower():
                        learning_style = 'hands_on'
                    elif 'observe' in q6_answer.lower() or 'watch' in q6_answer.lower():
                        learning_style = 'watch_first'
                    elif 'mix' in q6_answer.lower() or 'both' in q6_answer.lower():
                        learning_style = 'hybrid'
                    elif 'ease' in q6_answer.lower() or 'slow' in q6_answer.lower() or 'gradual' in q6_answer.lower():
                        learning_style = 'exploratory'

                    # Map last_theme to intensity
                    last_theme = act_2_state.get('last_theme', '')
                    if last_theme == 'tight_fit':
                        intensity = 'compact'
                    elif last_theme in ['flexible_fit', 'variable_fit']:
                        intensity = 'immersive'

                    # Build subfolder name or choose randomly
                    if learning_style and intensity:
                        subfolder_name = f"{learning_style}__{intensity}"
                        q10_image_folder = os.path.join(q10_base_folder, subfolder_name)
                        print(f"📁 Determined Q10 subfolder: {subfolder_name}")
                    else:
                        # Choose randomly from available subfolders
                        import glob
                        available_subfolders = [d for d in glob.glob(os.path.join(q10_base_folder, "*")) if
                                                os.path.isdir(d)]
                        if available_subfolders:
                            q10_image_folder = random.choice(available_subfolders)
                            print(f"🎲 Randomly selected Q10 subfolder: {os.path.basename(q10_image_folder)}")
                        else:
                            q10_image_folder = q10_base_folder
                            print(f"⚠️ No subfolders found, using base folder")

                    if os.path.exists(q10_image_folder):
                        # Get all PNG images in the selected folder
                        import glob
                        image_files = sorted(glob.glob(os.path.join(q10_image_folder, "*.png")))
                        if image_files:
                            # Populate options with full image paths
                            options = image_files
                            # Populate mappings with image names (without extension)
                            act_2_emotional_mapping = [os.path.splitext(os.path.basename(img))[0] for img in image_files]
                            print(f"✅ Loaded {len(options)} images for Act 2 Q10 from {os.path.basename(q10_image_folder)}")
                        else:
                            print(f"⚠️ No PNG images found in {q10_image_folder}")
                    else:
                        print(f"⚠️ Image folder not found: {q10_image_folder}")

                # DEBUG: Check what we're about to use
                print(f"DEBUG Q10 - About to create response with:")
                print(f"  current_turn: {current_turn}")
                print(f"  options: {options}")
                print(f"  act_2_emotional_mapping: {act_2_emotional_mapping}")

                # THIS MUST BE AT THE SAME INDENTATION LEVEL AS THE "if current_turn == 4:" - NOT INSIDE IT!
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
                # Cache the response
                raw_response = get_cached_llm_response(self.name, current_turn, structured_llm, msgs)
                strict_response: MotivationResponse = raw_response

                # Build metadata from optional fields
                metadata = {}
                if strict_response.response_format:
                    metadata['response_format'] = strict_response.response_format
                if strict_response.scale_range:
                    metadata['scale_range'] = strict_response.scale_range
                if strict_response.scale_labels:
                    metadata['scale_labels'] = strict_response.scale_labels
                if strict_response.scale_mapping:
                    metadata['scale_mapping'] = strict_response.scale_mapping

                # ===== FIX: For scale questions, clear options and mapping =====
                options = strict_response.options or []
                option_mapping = strict_response.act_3_mapping or []

                if strict_response.response_format == "scale":
                    # Scale questions don't use options/mapping
                    options = []
                    option_mapping = []

                # Convert to AgentResponse format - handle None values
                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=options,
                    option_mapping=option_mapping,
                    metadata=metadata,
                    state={}
                )

            elif self.info_type == "barriers":
                structured_llm = llm.with_structured_output(BarriersResponse)
                # Cache the response
                raw_response = get_cached_llm_response(self.name, current_turn, structured_llm, msgs)
                strict_response: BarriersResponse = raw_response

                # Build metadata from optional fields
                metadata = {}
                if strict_response.response_format:
                    metadata['response_format'] = strict_response.response_format
                if strict_response.scale_range:
                    metadata['scale_range'] = strict_response.scale_range
                if strict_response.scale_labels:
                    metadata['scale_labels'] = strict_response.scale_labels
                if strict_response.scale_mapping:
                    metadata['scale_mapping'] = strict_response.scale_mapping

                # ===== FIX: For scale questions, clear options and mapping =====
                options = strict_response.options or []
                option_mapping = strict_response.act_4_mapping or []

                if strict_response.response_format == "scale":
                    # Scale questions don't use options/mapping
                    # Pad with 4 empty strings to match Act 2/3 behavior
                    options = ['', '', '', '']
                    option_mapping = ['', '', '', '']

                # Convert to AgentResponse format
                response = AgentResponse(
                    affirmation=strict_response.affirmation,
                    question_text=strict_response.question_text,
                    options=options,
                    option_mapping=option_mapping,
                    metadata=metadata,
                    state={}
                )

            # Use strict schema for summary agent
            elif self.info_type == "summary":
                structured_llm = llm.with_structured_output(SummaryResponse)
                # Cache the response
                raw_response = get_cached_llm_response(self.name, current_turn, structured_llm, msgs)
                strict_response: SummaryResponse = raw_response
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
                # Use random selection for courses agent (no LLM needed)

            # Use random selection for courses agent (no LLM needed)
            elif self.info_type == "courses":
                # Randomly select 5 courses using existing helper function
                selected_courses = select_learning_screens()

                response = AgentResponse(
                    question_text="",
                    options=[],
                    option_mapping=["", "", "", ""],
                    metadata={
                        "selected_courses": selected_courses,
                        "selection_reasoning": "Random selection"
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

            # CHANGE 9: Return proper fallback responses for connection_intent agent
            if self.info_type == "connection_intent":
                # Determine which question we're on
                stage_meta = state.get("stage_meta", {}) or {}
                act_1_state = stage_meta.get("act_1", {}).get("state", {})
                current_turn = act_1_state.get("turn", 0)
                question_num = current_turn + 1

                # Provide fallback based on question number
                if question_num == 1:
                    # Q1 fallback - multiple choice
                    return AgentResponse(
                        affirmations=["I'd love to understand what brings you here today.",
                                      "I'd love to understand what brings you here today.",
                                      "I'd love to understand what brings you here today.",
                                      "I'd love to understand what brings you here today."],
                        question_text="When I think about creative work, I most want to feel like...",
                        options=[
                            "Someone who finishes what they start",
                            "Someone with a real creative hobby",
                            "Someone seen as creative",
                            "I'm not quite sure yet"
                        ],
                        option_mapping=["finishes_what_they_start", "real_creative_hobby", "seen_as_creative",
                                        "unsure"],
                        metadata={
                            "error": str(e),
                            "response_format": "multiple_choice",
                            "nested_option_mapping": [
                                {"identity_shift_cluster": "finishes_what_they_start",
                                 "aspiration_category": "confidence_progress"},
                                {"identity_shift_cluster": "real_creative_hobby",
                                 "aspiration_category": "enrichment_purpose"},
                                {"identity_shift_cluster": "seen_as_creative",
                                 "aspiration_category": "self_expression"},
                                {"identity_shift_cluster": "unsure", "aspiration_category": "unsure"}
                            ]
                        },
                        state={},
                    )
                elif question_num == 2:
                    # Q2 fallback - scale
                    return AgentResponse(
                        affirmations=["That's a meaningful desire.", "That's a meaningful desire.",
                                      "That's a meaningful desire.", "That's a meaningful desire."],
                        question_text="On a scale of 1-5, how close do you feel to this identity right now?",
                        options=[],
                        option_mapping=["", "", "", ""],
                        metadata={
                            "error": str(e),
                            "response_format": "scale",
                            "scale_range": "1-5",
                            "scale_labels": {"min": "Not close at all", "max": "Very close"},
                            "scale_mapping": {"1-2": "low", "3": "medium", "4-5": "high"}
                        },
                        state={},
                    )
                elif question_num == 3:
                    # Q3 fallback - image select
                    return AgentResponse(
                        affirmations=["Let's explore what resonates with you visually.",
                                      "Let's explore what resonates with you visually.",
                                      "Let's explore what resonates with you visually.",
                                      "Let's explore what resonates with you visually."],
                        question_text="Which image speaks to you most?",
                        options=[],
                        option_mapping=["", "", "", ""],
                        metadata={"error": str(e), "response_format": "image_select"},
                        state={},
                    )
                elif question_num == 4:
                    # Q4 fallback - scale
                    return AgentResponse(
                        affirmations=["Your feelings are valid.", "Your feelings are valid.",
                                      "Your feelings are valid.", "Your feelings are valid."],
                        question_text="How much does your current life situation support this identity shift?",
                        options=[],
                        option_mapping=["", "", "", ""],
                        metadata={
                            "error": str(e),
                            "response_format": "scale",
                            "scale_range": "1-5",
                            "scale_labels": {"min": "Not supportive", "max": "Very supportive"},
                            "scale_mapping": {"1-2": "low_support", "3": "neutral", "4-5": "high_support"}
                        },
                        state={},
                    )
                elif question_num == 5:
                    # Q5 fallback - either_or
                    return AgentResponse(
                        affirmations=["This is the final piece.", "This is the final piece.",
                                      "This is the final piece.", "This is the final piece."],
                        question_text="Which feels more true for you right now?",
                        options=[
                            "This identity shift feels aligned with who I want to become",
                            "This identity shift feels forced or misaligned"
                        ],
                        option_mapping=["aligned", "misaligned", "", ""],
                        metadata={
                            "error": str(e),
                            "response_format": "either_or",
                            "nested_option_mapping": [
                                {"alignment": "aligned"},
                                {"alignment": "misaligned"}
                            ]
                        },
                        state={},
                    )
                else:
                    # Generic fallback
                    return AgentResponse(
                        affirmations=["I'd love to understand what brings you here today.",
                                      "I'd love to understand what brings you here today.",
                                      "I'd love to understand what brings you here today.",
                                      "I'd love to understand what brings you here today."],
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
                    affirmations=["It's completely okay to feel however you're feeling right now.",
                                  "It's completely okay to feel however you're feeling right now.",
                                  "It's completely okay to feel however you're feeling right now.",
                                  "It's completely okay to feel however you're feeling right now."],
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
                    affirmations=["Understanding what truly drives you is so important.",
                                  "Understanding what truly drives you is so important.",
                                  "Understanding what truly drives you is so important.",
                                  "Understanding what truly drives you is so important."],
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
                    affirmations=["It's important to understand what might be in your way.",
                                  "It's important to understand what might be in your way.",
                                  "It's important to understand what might be in your way.",
                                  "It's important to understand what might be in your way."],
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

        # CHANGE 10: Track conversation history (last 4 turns)
        conversation_history = list(state.get("conversation_history", []))
        # Don't add demographics or hook to conversation history
        if self.info_type not in ["hook", "demographics"]:
            conversation_turn = {
                "affirmation": "",  # Will be filled later when user selects an answer
                "question": response.question_text or "",
                "answer": "",  # Will be filled when user responds
                "options": response.options or [],
                "affirmations": response.get_affirmations()  # Store all affirmations for later matching
            }
        else:
            conversation_turn = None



        # Build user-facing text: affirmation + question + enumerated options
        display_text = ""



        # Check if this is from hook agent and format accordingly
        if self.info_type == "hook" and response.question_text:
            display_text += format_hook_text(response.question_text)
        elif response.question_text:
            display_text += format_question_text(response.question_text)

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

        resp_meta = dict(response.metadata or {})
        resp_state = dict(response.state or {})

        new_meta = dict(old_meta)
        new_state = dict(old_state)

        # Merge metadata first (authoritative for rendering contracts)
        if resp_meta:
            new_meta.update(resp_meta)

        # Merge state, BUT never allow scale fields to come from response.state
        if resp_state:
            response_format = resp_meta.get("response_format")
            for k, v in resp_state.items():
                if response_format == "scale" and k in (
                        "scale_range",
                        "scale_labels",
                        "scale_mapping",
                ):
                    continue  # 🔥 block overwrite
                new_state[k] = v

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
                # For scale: store scale_range, scale_labels, and scale_mapping
                new_state["scale_range"] = resp_meta.get("scale_range", "")
                new_state["scale_labels"] = resp_meta.get("scale_labels", {})
                new_state["scale_mapping"] = resp_meta.get("scale_mapping", {})
                # Don't store options/option_mapping for scales
            else:
                # For multiple_choice, either_or, and image_choice: store options and mapping
                if response.option_mapping:
                    new_state["option_mapping"] = response.option_mapping
                if response.options:
                    new_state["options"] = response.options
                affirmations = response.get_affirmations()
                if affirmations:
                    new_state["affirmations"] = affirmations

            # Store image URLs if present (for image_choice format)
            if resp_meta.get("image_urls"):
                new_state["image_urls"] = resp_meta.get("image_urls", [])

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
                # Initialize 6 answer fields for Q6-Q11
                new_state["act_2_answer_1"] = ""
                new_state["act_2_answer_2"] = ""
                new_state["act_2_answer_3"] = ""
                new_state["act_2_answer_4"] = ""
                new_state["act_2_answer_5"] = ""
                new_state["act_2_answer_6"] = ""

                # Initialize response formats for 6 questions
                new_state["response_format_1"] = ""
                new_state["response_format_2"] = ""
                new_state["response_format_3"] = ""
                new_state["response_format_4"] = ""
                new_state["response_format_5"] = ""
                new_state["response_format_6"] = ""
                new_state["last_theme"] = ""

            # Get response format from metadata
            response_format = resp_meta.get("response_format", "multiple_choice")

            # Store response format for this turn (6 questions)
            if current_turn == 1:
                new_state["response_format_1"] = response_format
            elif current_turn == 2:
                new_state["response_format_2"] = response_format
            elif current_turn == 3:
                new_state["response_format_3"] = response_format
            elif current_turn == 4:
                new_state["response_format_4"] = response_format
            elif current_turn == 5:
                new_state["response_format_5"] = response_format
            elif current_turn == 6:
                new_state["response_format_6"] = response_format
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
                affirmations = response.get_affirmations()
                if affirmations:
                    new_state["affirmations"] = affirmations

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

            # 🔥 NEW: handle response format like Act 2
            response_format = resp_meta.get("response_format", "multiple_choice")
            new_state["response_format"] = response_format
            if response_format == "scale":
                new_state["scale_range"] = resp_meta.get("scale_range", "")
                new_state["scale_labels"] = resp_meta.get("scale_labels", {})
                new_state["scale_mapping"] = resp_meta.get("scale_mapping", {})
            else:
                if response.option_mapping:
                    new_state["act_3_mapping"] = response.option_mapping
                if response.options:
                    new_state["options"] = response.options
                    affirmations = response.get_affirmations()
                    if affirmations:
                        new_state["affirmations"] = affirmations

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
                new_state["act_4_answer_3"] = ""
                new_state["act_4_answer_4"] = ""
                new_state["last_act_4"] = ""  # Will be set after first answer

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
                new_state["last_level"] = "L4"  # 4 turns total

            # Store act_4_mapping and options from LLM response
            if response.option_mapping:
                new_state["act_4_mapping"] = response.option_mapping
            if response.options:
                new_state["options"] = response.options
                affirmations = response.get_affirmations()
                if affirmations:
                    new_state["affirmations"] = affirmations

            # Store response_format and scale fields from metadata
            if response.metadata:
                if "response_format" in response.metadata:
                    new_state["response_format"] = response.metadata["response_format"]
                if "scale_range" in response.metadata:
                    new_state["scale_range"] = response.metadata["scale_range"]
                if "scale_labels" in response.metadata:
                    new_state["scale_labels"] = response.metadata["scale_labels"]
                if "scale_mapping" in response.metadata:
                    new_state["scale_mapping"] = response.metadata["scale_mapping"]

            # Keep metadata fields (these will be updated by update_act_4_metadata_after_answer)
            new_meta.setdefault("confirm_act_4", "unclear")
            new_meta.setdefault("act_4_type", "")
            new_state.setdefault("last_act_4", "")

            stage_meta_prev[self.name] = {
                "metadata": new_meta,
                "state": new_state,
            }
        # ---------- COURSES AGENT LOGIC ----------
        if self.info_type == "courses":
            # Courses runs only once (turn = 1)
            new_state["turn"] = 1

            # Store selected courses and reasoning from response
            selected_courses = response.metadata.get("selected_courses", [])
            selection_reasoning = response.metadata.get("selection_reasoning", "")

            # Update global state with selected courses
            new_state["selected_courses"] = selected_courses
            new_state["selection_reasoning"] = selection_reasoning

            # Store in metadata
            new_meta["selected_courses"] = selected_courses
            new_meta["selection_reasoning"] = selection_reasoning
            new_meta["courses_turn"] = 1

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

        # Add current turn to conversation history and keep only last 4
        if conversation_turn is not None:
            conversation_history.append(conversation_turn)
            conversation_history = conversation_history[-4:]  # Keep only last 4 turns

        return AgentState(
            messages=[AIMessage(content=display_text)],
            collected_info=merged_collected,
            stage_meta=stage_meta_prev,
            exchanges_with_current=state.get("exchanges_with_current", 0) + 1,
            last_agent=self.name,
            conversation_history=conversation_history,
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
        "connection_intent": ConnectionAgent,  # new intent agent
        "connection_tone": ConnectionAgent,  # new emotional tone agent

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

        # CHANGE 8: Updated completion check for Q1-Q5 format
        # Act 1 is complete when confirm_act_1 == "clear" (after Q5)
        act_1_complete = (act_1_status == "clear")

        # Act 2 is complete when confirm_act_2 == "clear"
        act_2_complete = (act_2_emo_tone not in ("resistant", "unclear"))

        # Stage 1 is ready when Act 1 is complete
        ready = act_1_complete and act_2_complete

        # --- If ready for deep stage or we've hit max turns, route out of Stage 1 ---
        if ready or total_turns >= 10:  # Increased from 6 to 10 for Q1-Q5 format
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

        # Buttons/Input for current options (only if conversation is ongoing)
        options = st.session_state.state.get("last_options", []) or []
        # Extract demographics status
        demo_block = stage_meta.get("demographics", {}) or {}
        demo_meta = demo_block.get("metadata", {}) or {}
        demo_status = demo_meta.get("demo_status", "not_started")

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

        # Extract courses info
        courses_block = stage_meta.get("courses", {}) or {}
        courses_meta = courses_block.get("metadata", {}) or {}

        courses_turn = courses_meta.get("courses_turn", 0)
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
            # Demographics variables
            "user_age": state.get("user_age", "") or "",
            "user_gender": state.get("user_gender", "") or "",
            "demo_status": demo_status,
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
            "courses_turn": str(courses_turn),
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
            # Use cached LLM instead of creating new instance
            llm = get_cached_llm()
            if llm is None:
                raise Exception("LLM not available")
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

        # If LLM suggested an unknown agent (except demographics), fallback to deterministic routing
        if next_agent not in self.agent_keys and next_agent != "demographics":
            print(f"DEBUG SUPERVISOR LLM - unknown agent '{next_agent}', falling back to route()")
            return self.route(state)
        # Valid agent → return routing info
        # For demographics node, don't set agent_index (it's not in agent_keys)
        if next_agent == "demographics":
            return {
                "next_agent": next_agent,
                "exchanges_with_current": 0,
            }
        else:
            return {
                "next_agent": next_agent,
                "agent_index": self.agent_keys.index(next_agent),
                "exchanges_with_current": 0,
            }


# ============================================================================
# GRAPH CREATION
# ============================================================================
def demographics_node(state: AgentState) -> AgentState:
    """
    Hardcoded demographics collection node.
    Asks for age, then gender, then marks complete.
    """
    stage_meta = state.get("stage_meta", {}) or {}

    # Initialize demographics metadata if not exists
    if "demographics" not in stage_meta:
        stage_meta["demographics"] = {
            "metadata": {"demo_status": "not_started", "collected_age": "", "collected_gender": ""},
            "state": {}
        }

    demo_meta = stage_meta["demographics"]["metadata"]
    collected_age = demo_meta.get("collected_age", "")
    collected_gender = demo_meta.get("collected_gender", "")

    user_input = state.get("user_input", "")

    print(f"DEBUG demographics_node - START:")
    print(f"  user_input: '{user_input}'")
    print(f"  collected_age: '{collected_age}'")
    print(f"  collected_gender: '{collected_gender}'")

    # Determine what to do based on what's been collected
    if not collected_age:
        # Need to collect age
        if user_input and user_input in ["18-24", "25-34", "35-44", "45-54", "55+"]:
            # User just answered age question - store it and ask gender
            demo_meta["collected_age"] = user_input
            demo_meta["demo_status"] = "asking_gender"

            print(f"DEBUG - Stored age: {user_input}, now asking gender")

            # Ask gender question
            question_text = "What is your gender?"
            options = ["Male", "Female", "Other"]
            display_text = f"<div style='text-align: center; font-size: 36px;'>{question_text}</div>\n\n"
            display_text += "A) Male\n"
            display_text += "B) Female\n"
            display_text += "C) Other"

            return {
                **state,
                "messages": [AIMessage(content=display_text)],
                "stage_meta": stage_meta,
                "last_agent": "demographics",
                "last_options": options,
            }
        else:
            # Ask age question
            question_text = "What is your age?"
            options = ["18-24", "25-34", "35-44", "45-54", "55+"]

            print(f"DEBUG - Asking for age")

            display_text = f"<div style='text-align: center; font-size: 36px;'>{question_text}</div>\n\n"
            display_text += "A) 18-24\n"
            display_text += "B) 25-34\n"
            display_text += "C) 35-44\n"
            display_text += "D) 45-54\n"
            display_text += "E) 55+"

            return {
                **state,
                "messages": [AIMessage(content=display_text)],
                "stage_meta": stage_meta,
                "last_agent": "demographics",
                "last_options": options,
            }

    elif not collected_gender:
        # Need to collect gender
        if user_input and user_input in ["Male", "Female", "Other"]:
            # User just answered gender question - mark complete
            demo_meta["collected_gender"] = user_input
            demo_meta["demo_status"] = "complete"

            print(f"DEBUG - Stored gender: {user_input}, demographics complete")

            return {
                **state,
                "stage_meta": stage_meta,
                "last_agent": "demographics",
                "last_options": [],
                "user_age": collected_age,
                "user_gender": user_input,
            }
        else:
            # Ask gender question
            question_text = "What is your gender?"
            options = ["Male", "Female", "Other"]

            print(f"DEBUG - Asking for gender")

            display_text = f"<div style='text-align: center; font-size: 36px;'>{question_text}</div>\n\n"
            display_text += "A) Male\n"
            display_text += "B) Female\n"
            display_text += "C) Other"

            return {
                **state,
                "messages": [AIMessage(content=display_text)],
                "stage_meta": stage_meta,
                "last_agent": "demographics",
                "last_options": options,
            }

    else:
        # Both collected, mark complete
        print(f"DEBUG - Demographics already complete (age={collected_age}, gender={collected_gender})")

        return {
            **state,
            "stage_meta": stage_meta,
            "last_agent": "demographics",
            "last_options": [],
            "user_age": collected_age,
            "user_gender": collected_gender,
        }


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
        # Add demographics node (hardcoded, not an agent)
    workflow.add_node("demographics", demographics_node)
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
        # Handle demographics routing
        if nxt == "demographics":
            print("DEBUG ROUTER - Routing to demographics node")
            return "demographics"
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
    edges_map.update({"demographics": "demographics"})  # ADD THIS LINE
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

    # Demographics also ends after running (waits for user input)  # ADD THIS LINE
    workflow.add_edge("demographics", END)  # ADD THIS LINE

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
    # Update conversation history with user's answer to previous question
    if msg:
        conversation_history = list(state.get("conversation_history", []))
        if conversation_history and isinstance(conversation_history[-1], dict) and not conversation_history[-1].get(
                "answer"):
            conversation_history[-1]["answer"] = msg
            state = {**state, "conversation_history": conversation_history}

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
        # Update conversation_history with semantic context for scale/image questions
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            last_turn = conversation_history[-1]
            act_1_state = state.get("stage_meta", {}).get("act_1", {}).get("state", {})
            current_turn = act_1_state.get("turn", 0)

            # For Q2 (scale - presence), replace raw number with semantic value
            if current_turn == 2:
                presence = act_1_state.get("q2_presence", "")
                if presence:
                    last_turn["answer"] = f"{msg} (presence level: {presence})"

            # For Q3 (image), add semantic mapping
            elif current_turn == 3:
                mapping = act_1_state.get("q3_mapping", "")
                facet = act_1_state.get("q3_facet", "")
                if mapping or facet:
                    last_turn["answer"] = f"Selected image: {mapping} ({facet})"

            # For Q4 (scale - context influence), replace raw number with semantic value
            elif current_turn == 4:
                context = act_1_state.get("q4_context", "")
                if context:
                    last_turn["answer"] = f"{msg} (context influence: {context})"

            # Update state with modified conversation_history
            state = {**state, "conversation_history": conversation_history}

        print(
            f"DEBUG - After intent metadata update: confirm_act_1={state.get('stage_meta', {}).get('act_1', {}).get('metadata', {}).get('confirm_act_1')}")

    # Update emotional tone metadata if user answered an emotional tone question
    if last_agent == "act_2" and msg:
        state = update_act_2_metadata_after_answer(state, msg)
        # Update conversation_history with semantic context for Act 2
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            last_turn = conversation_history[-1]
            act_2_state = state.get("stage_meta", {}).get("act_2", {}).get("state", {})
            current_turn = act_2_state.get("turn", 0)

            # Add semantic values for scale questions (Q9 and Q11)
            if current_turn == 4:  # Q9 - Time/Energy Fit scale
                time_energy = act_2_state.get("time_energy_fit", "")
                if time_energy:
                    last_turn["answer"] = f"{msg} (time/energy fit: {time_energy})"
            elif current_turn == 6:  # Q11 - Emotional Safety scale
                safety = act_2_state.get("emotional_safety_level", "")
                if safety:
                    last_turn["answer"] = f"{msg} (emotional safety: {safety})"

            state = {**state, "conversation_history": conversation_history}

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

    # Check if act_4 just completed - if so, trigger course selection
    act_4_block = state.get("stage_meta", {}).get("act_4", {})
    act_4_status = act_4_block.get("metadata", {}).get("confirm_act_4", "unclear")

    if act_4_status == "clear" and not state.get("course_selection_complete", False):
        # Act 4 just completed and courses haven't been selected yet
        if not state.get("selected_courses"):
            # Select courses for the user
            selected_courses = select_courses_for_user(state)
            state = {**state, "selected_courses": selected_courses}
            print(f"✅ Course selection triggered. Selected courses: {selected_courses}")
            # Return empty response - UI will show course selection screens
            return "", state

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
# CACHING FUNCTIONS - Solution 2: Cache Sidebar Data Extraction
# ============================================================================

@st.cache_data
def extract_sidebar_data(_state: AgentState) -> Dict[str, Any]:
    """
    Extract and cache sidebar data to avoid recalculating on every rerun.
    Only recalculates when state actually changes.

    Savings: 50-100ms per rerun
    """
    stage_meta = _state.get("stage_meta", {}) or {}

    # Demographics
    demo_block = stage_meta.get("demographics", {}) or {}
    demo_meta = demo_block.get("metadata", {}) or {}
    collected_age = demo_meta.get("collected_age", "Not set")
    collected_gender = demo_meta.get("collected_gender", "Not set")

    # Fallback to state values
    if collected_age == "Not set":
        collected_age = _state.get("user_age", "Not set")
    if collected_gender == "Not set":
        collected_gender = _state.get("user_gender", "Not set")

    # Hook
    hook_block = stage_meta.get("hook", {}) or {}
    hook_meta = hook_block.get("metadata", {}) or {}
    hook_status = hook_meta.get("hook_status", "unclear")
    hook_text = hook_meta.get("hook_text", "—")

    # Act 1
    act_1_block = stage_meta.get("act_1", {}) or {}
    act_1_meta = act_1_block.get("metadata", {}) or {}
    act_1_state = act_1_block.get("state", {}) or {}
    act_1_status = act_1_meta.get("confirm_act_1", "—")
    act_1_type = act_1_meta.get("act_1_type", "—")
    current_turn = act_1_state.get("turn", 0)
    last_theme = act_1_state.get("last_theme", "—")
    theme_1 = act_1_state.get("theme_1", "—")
    theme_2 = act_1_state.get("theme_2", "—")
    theme_3 = act_1_state.get("theme_3", "—")
    theme_4 = act_1_state.get("theme_4", "—")

    # Question mode/focus type calculation
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

    # Act 2
    act_2_block = stage_meta.get("act_2", {}) or stage_meta.get("connection_tone", {}) or {}
    act_2_meta = act_2_block.get("metadata", {}) or {}
    act_2_state = act_2_block.get("state", {}) or {}
    confirm_act_2 = act_2_meta.get("confirm_act_2", "unclear")
    act_2_type = act_2_meta.get("act_2_emo_tone") or act_2_meta.get("emo_act_2_type") or "—"
    # Act 2 - 6 answers (Q6-Q11)
    act_2_answer_1 = act_2_state.get("act_2_answer_1", "—")
    act_2_answer_2 = act_2_state.get("act_2_answer_2", "—")
    act_2_answer_3 = act_2_state.get("act_2_answer_3", "—")
    act_2_answer_4 = act_2_state.get("act_2_answer_4", "—")
    act_2_answer_5 = act_2_state.get("act_2_answer_5", "—")
    act_2_answer_6 = act_2_state.get("act_2_answer_6", "—")

    # Act 2 - Derived learning behavior signals
    entry_style = act_2_state.get("entry_style", "—")
    momentum_support = act_2_state.get("momentum_support", "—")
    situational_friction = act_2_state.get("situational_friction", "—")
    time_energy_fit = act_2_state.get("time_energy_fit", "—")
    learning_mode = act_2_state.get("learning_mode", "—")
    emotional_safety_level = act_2_state.get("emotional_safety_level", "—")

    # Act 2 turn info
    act_2_turn = act_2_state.get("turn", 0)
    act_2_last_theme = act_2_state.get("last_theme", "—")

    # Act 3
    act_3_block = stage_meta.get("act_3", {}) or {}
    act_3_meta = act_3_block.get("metadata", {}) or {}
    act_3_state = act_3_block.get("state", {}) or {}
    confirm_act_3 = act_3_meta.get("confirm_act_3", "unclear")
    act_3_type = act_3_meta.get("act_3_type", "—")
    act_3_answer_1 = act_3_state.get("act_3_answer_1", "—")
    act_3_answer_2 = act_3_state.get("act_3_answer_2", "—")
    act_3_answer_3 = act_3_state.get("act_3_answer_3", "—")
    act_3_answer_4 = act_3_state.get("act_3_answer_4", "—")

    # Act 3 turn info
    act_3_turn = act_3_state.get("turn", 0)
    act_3_last_theme = act_3_state.get("last_theme", "—")

    # Compute Act 3 question mode
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

    # Act 4
    act_4_block = stage_meta.get("act_4", {}) or {}
    act_4_meta = act_4_block.get("metadata", {}) or {}
    act_4_state = act_4_block.get("state", {}) or {}
    confirm_act_4 = act_4_meta.get("confirm_act_4", "unclear")
    act_4_type = act_4_meta.get("act_4_type", "—")
    act_4_answer_1 = act_4_state.get("act_4_answer_1", "—")
    act_4_answer_2 = act_4_state.get("act_4_answer_2", "—")

    # Act 4 turn info
    act_4_turn = act_4_state.get("turn", 0)
    act_4_last_theme = act_4_state.get("last_theme", "—")

    # Compute Act 4 question mode (4 questions)
    if act_4_turn == 0:
        act_4_question_mode = "—"
        act_4_focus_type = "—"
    elif act_4_turn == 1:
        act_4_question_mode = "broad"
        act_4_focus_type = "support"
    elif act_4_turn == 2:
        act_4_question_mode = "focused"
        act_4_focus_type = "support"
    elif act_4_turn == 3:
        act_4_question_mode = "broad"
        act_4_focus_type = "support"
    elif act_4_turn == 4:
        act_4_question_mode = "focused"
        act_4_focus_type = "support"
    else:
        act_4_question_mode = "—"
        act_4_focus_type = "—"

    return {
        # Demographics
        "collected_age": collected_age,
        "collected_gender": collected_gender,
        # Hook
        "hook_status": hook_status,
        "hook_text": hook_text,
        # Act 1
        "act_1_status": act_1_status,
        "act_1_type": act_1_type,
        "current_turn": current_turn,
        "last_theme": last_theme,
        "question_mode": question_mode,
        "focus_type": focus_type,
        "theme_1": theme_1,
        "theme_2": theme_2,
        "theme_3": theme_3,
        "theme_4": theme_4,
        # Act 2
        "confirm_act_2": confirm_act_2,
        "act_2_type": act_2_type,
        # Act 2
        "confirm_act_2": confirm_act_2,
        "act_2_answer_1": act_2_answer_1,
        "act_2_answer_2": act_2_answer_2,
        "act_2_answer_3": act_2_answer_3,
        "act_2_answer_4": act_2_answer_4,
        "act_2_answer_5": act_2_answer_5,
        "act_2_answer_6": act_2_answer_6,
        "entry_style": entry_style,
        "momentum_support": momentum_support,
        "situational_friction": situational_friction,
        "time_energy_fit": time_energy_fit,
        "learning_mode": learning_mode,
        "emotional_safety_level": emotional_safety_level,
        "act_2_turn": act_2_turn,
        "act_2_last_theme": act_2_last_theme,
        # Act 3
        "confirm_act_3": confirm_act_3,
        "act_3_type": act_3_type,
        "act_3_answer_1": act_3_answer_1,
        "act_3_answer_2": act_3_answer_2,
        "act_3_answer_3": act_3_answer_3,
        "act_3_answer_4": act_3_answer_4,
        "act_3_turn": act_3_turn,
        "act_3_last_theme": act_3_last_theme,
        "act_3_question_mode": act_3_question_mode,
        "act_3_focus_type": act_3_focus_type,
        # Act 4
        "confirm_act_4": confirm_act_4,
        "act_4_type": act_4_type,
        "act_4_answer_1": act_4_answer_1,
        "act_4_answer_2": act_4_answer_2,
        "act_4_turn": act_4_turn,
        "act_4_last_theme": act_4_last_theme,
        "act_4_question_mode": act_4_question_mode,
        "act_4_focus_type": act_4_focus_type,
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

def _init_session():
    if "graph" not in st.session_state:
        st.session_state.graph = get_cached_graph()  # ✅ Use cached version

    if "ad_data" not in st.session_state:
        st.session_state.ad_data = get_ad_data_from_external_source()

    if "current_question_key" not in st.session_state:
        st.session_state.current_question_key = None

    if "current_image_path" not in st.session_state:
        st.session_state.current_image_path = "Act1/Q1/inspiration.png"

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
                },
                "act_1": {
                    "metadata": {
                        "confirm_act_1": "unclear",
                        "act_1_type": ""
                    },
                    "state": {
                        "turn": 0,  # Will go 1→5 for Q1-Q5
                        "selected_identity_cluster": "",
                        "mapped_aspiration_category": "",
                        "presence_level": "",
                        "context_influence": "",
                        "alignment_status": "",
                        # Store each question's data
                        "q1_answer": "",
                        "q1_identity": "",
                        "q1_aspiration": "",
                        "q2_answer": 0,
                        "q2_presence": "",
                        "q3_answer": "",
                        "q3_mapping": "",
                        "q3_facet": "",
                        "q4_answer": 0,
                        "q4_context": "",
                        "q5_answer": "",
                        "q5_alignment": "",
                        # Store image options for Q3
                        "last_image_mappings": [],
                        "last_image_paths": []
                    }
                }
            },
            last_options=[],

            ad_data=st.session_state.ad_data,
            user_profile={},
            conversation_history=[],
            user_age="",
            user_gender="",
            selected_courses=[],
            course_responses={},
            course_selection_complete=False,
        )

        # ✅ make sure connection_tone + emo_act_2_type + confirm_act_2 exist from the start
        base_state = ensure_act_2_block(base_state)

        st.session_state.state = base_state

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # UI Page State for 2-page flow (question page vs affirmation page)
    if "ui_page" not in st.session_state:
        st.session_state.ui_page = "question"  # "question" or "affirmation"

    # Preloaded next question for instant display
    if "next_question_preloaded" not in st.session_state:
        st.session_state.next_question_preloaded = None

    if "next_state_preloaded" not in st.session_state:
        st.session_state.next_state_preloaded = None

    # Current matched affirmation to display
    if "current_matched_affirmation" not in st.session_state:
        st.session_state.current_matched_affirmation = ""

    # Store last affirmations array for matching
    if "last_affirmations" not in st.session_state:
        st.session_state.last_affirmations = []
    # Course selection tracking
    if "current_course_index" not in st.session_state:
        st.session_state.current_course_index = 0

@st.cache_resource
def preload_graph():
    """Pre-load the graph structure into cache"""
    import time
    start = time.time()
    graph = get_cached_graph()
    elapsed = (time.time() - start) * 1000
    print(f"✅ Graph pre-loaded and cached: {elapsed:.0f}ms")
    return graph


# Call it to trigger caching
_ = preload_graph()


def main():
    st.set_page_config(page_title="Sparky – Enhanced Multi-Stage", layout="wide")
    # Custom CSS to make buttons much bigger
    st.markdown("""
            <style>
            div.stButton > button * {
                font-size: 20px !important;
            }
            div.stButton > button {
                font-size: 20px !important;
                padding: 20px 40px !important;
                height: auto !important;
                min-height: 70px !important;
            }
            </style>
        """, unsafe_allow_html=True)

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
        # Display user demographics
        st.header("👤 User Profile")

        # Extract all sidebar data using cached function (Solution 2)
        sidebar_data = extract_sidebar_data(st.session_state.state)

        st.markdown(f"**Age Range:** {sidebar_data['collected_age']}")
        st.markdown(f"**Gender:** {sidebar_data['collected_gender']}")
        st.markdown("---")
        st.markdown("---")
        st.header("Conversation Insights")

        # Hook block (CACHED)
        st.markdown("**🎣 Hook:**")
        st.markdown(f"- **Status:** {sidebar_data['hook_status']}")
        if sidebar_data['hook_text'] != "—":
            st.markdown(f"- **Message:** {sidebar_data['hook_text'][:100]}...")  # Show first 100 chars
        st.markdown("---")

        # Intent block (act_1) - CACHED
        meta = st.session_state.state.get("stage_meta", {}) or {}
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

        # Display insights (CACHED)
        st.markdown(f"- **Confirm Act 1:** {sidebar_data['act_1_status'] or '—'}")
        st.markdown(f"- **Act 1 Type:** {sidebar_data['act_1_type'] or '—'}")

        # Act 1 Details (CACHED)
        st.markdown("---")
        st.markdown("**🎯 Act 1 Details:**")
        st.markdown(f"- **Current Turn:** {sidebar_data['current_turn']}")
        st.markdown(f"- **Last Theme:** {sidebar_data['last_theme']}")
        st.markdown(f"- **Question Mode (current question):** {sidebar_data['question_mode']}")
        st.markdown(f"- **Focus Type (current question):** {sidebar_data['focus_type']}")

        # Show individual answers (as mappings) if available
        if any([sidebar_data['theme_1'] != "—", sidebar_data['theme_2'] != "—",
                sidebar_data['theme_3'] != "—", sidebar_data['theme_4'] != "—"]):
            st.markdown("**Answers:**")
            if sidebar_data['theme_1'] != "—":
                st.markdown(f"  - Q1: {sidebar_data['theme_1']}")
            if sidebar_data['theme_2'] != "—":
                st.markdown(f"  - Q2: {sidebar_data['theme_2']}")
            if sidebar_data['theme_3'] != "—":
                st.markdown(f"  - Q3: {sidebar_data['theme_3']}")
            if sidebar_data['theme_4'] != "—":
                st.markdown(f"  - Q4: {sidebar_data['theme_4']}")

        st.markdown("---")

        # Act 2 section (CACHED)
        st.markdown("**😊 Act 2 Details:**")
        st.markdown(f"- **Current Turn:** {sidebar_data['act_2_turn']}")
        st.markdown(f"- **Last Theme:** {sidebar_data['act_2_last_theme']}")
        st.markdown(f"- **Confirm Act 2:** {sidebar_data['confirm_act_2']}")

        # Show individual answers if available (6 questions: Q6-Q11)
        if any([sidebar_data.get('act_2_answer_1') != "—",
                sidebar_data.get('act_2_answer_2') != "—",
                sidebar_data.get('act_2_answer_3') != "—",
                sidebar_data.get('act_2_answer_4') != "—",
                sidebar_data.get('act_2_answer_5') != "—",
                sidebar_data.get('act_2_answer_6') != "—"]):
            st.markdown("**Answers:**")
            if sidebar_data.get('act_2_answer_1') != "—":
                st.markdown(f"  - Q6 (Entry Style): {sidebar_data.get('act_2_answer_1')}")
            if sidebar_data.get('act_2_answer_2') != "—":
                st.markdown(f"  - Q7 (Momentum): {sidebar_data.get('act_2_answer_2')}")
            if sidebar_data.get('act_2_answer_3') != "—":
                st.markdown(f"  - Q8 (Friction): {sidebar_data.get('act_2_answer_3')}")
            if sidebar_data.get('act_2_answer_4') != "—":
                st.markdown(f"  - Q9 (Time/Energy): {sidebar_data.get('act_2_answer_4')}")
            if sidebar_data.get('act_2_answer_5') != "—":
                st.markdown(f"  - Q10 (Learning Mode): {sidebar_data.get('act_2_answer_5')}")
            if sidebar_data.get('act_2_answer_6') != "—":
                st.markdown(f"  - Q11 (Safety): {sidebar_data.get('act_2_answer_6')}")

        # Show derived learning behavior signals
        st.markdown("**Derived Signals:**")
        if sidebar_data.get('entry_style') != "—":
            st.markdown(f"  - Entry Style: {sidebar_data.get('entry_style')}")
        if sidebar_data.get('momentum_support') != "—":
            st.markdown(f"  - Momentum: {sidebar_data.get('momentum_support')}")
        if sidebar_data.get('situational_friction') != "—":
            st.markdown(f"  - Friction: {sidebar_data.get('situational_friction')}")
        if sidebar_data.get('time_energy_fit') != "—":
            st.markdown(f"  - Time/Energy Fit: {sidebar_data.get('time_energy_fit')}")
        if sidebar_data.get('learning_mode') != "—":
            st.markdown(f"  - Learning Mode: {sidebar_data.get('learning_mode')}")
        if sidebar_data.get('emotional_safety_level') != "—":
            st.markdown(f"  - Safety Level: {sidebar_data.get('emotional_safety_level')}")

        st.markdown("---")

        # Act 3 section (CACHED)
        st.markdown("**💪 Act 3:**")
        st.markdown(f"- **Current Turn:** {sidebar_data['act_3_turn']}")
        st.markdown(f"- **Last Theme:** {sidebar_data['act_3_last_theme']}")
        st.markdown(f"- **Question Mode:** {sidebar_data['act_3_question_mode']}")
        st.markdown(f"- **Focus Type:** {sidebar_data['act_3_focus_type']}")
        st.markdown(f"- **Confirm Act 3:** {sidebar_data['confirm_act_3']}")
        st.markdown(f"- **Act 3 Type:** {sidebar_data['act_3_type']}")

        # Show individual answers if available
        if any([sidebar_data['act_3_answer_1'] != "—", sidebar_data['act_3_answer_2'] != "—",
                sidebar_data['act_3_answer_3'] != "—", sidebar_data['act_3_answer_4'] != "—"]):
            st.markdown("**Answers:**")
            if sidebar_data['act_3_answer_1'] != "—":
                st.markdown(f"  - Q1: {sidebar_data['act_3_answer_1']}")
            if sidebar_data['act_3_answer_2'] != "—":
                st.markdown(f"  - Q2: {sidebar_data['act_3_answer_2']}")
            if sidebar_data['act_3_answer_3'] != "—":
                st.markdown(f"  - Q3: {sidebar_data['act_3_answer_3']}")
            if sidebar_data['act_3_answer_4'] != "—":
                st.markdown(f"  - Q4: {sidebar_data['act_3_answer_4']}")

        st.markdown("---")

        # Act 4 section (CACHED)
        st.markdown("**🚧 Act 4:**")
        st.markdown(f"- **Current Turn:** {sidebar_data['act_4_turn']}")
        st.markdown(f"- **Last Theme:** {sidebar_data['act_4_last_theme']}")
        st.markdown(f"- **Question Mode:** {sidebar_data['act_4_question_mode']}")
        st.markdown(f"- **Focus Type:** {sidebar_data['act_4_focus_type']}")
        st.markdown(f"- **Confirm Act 4:** {sidebar_data['confirm_act_4']}")
        st.markdown(f"- **Act 4 Type:** {sidebar_data['act_4_type']}")

        # Show individual answers if available
        if any([sidebar_data['act_4_answer_1'] != "—", sidebar_data['act_4_answer_2'] != "—"]):
            st.markdown("**Answers:**")
            if sidebar_data['act_4_answer_1'] != "—":
                st.markdown(f"  - Q1: {sidebar_data['act_4_answer_1']}")
            if sidebar_data['act_4_answer_2'] != "—":
                st.markdown(f"  - Q2: {sidebar_data['act_4_answer_2']}")

        # Get stage_meta for the rest of the sidebar that needs original structure
        meta = st.session_state.state.get("stage_meta", {}) or {}
        act_1_block = meta.get("act_1", {}) or {}
        act_1_meta = act_1_block.get("metadata", {}) or {}
        act_1_state = act_1_block.get("state", {}) or {}
        act_2_block = meta.get("act_2", {}) or {}
        act_2_meta = act_2_block.get("metadata", {}) or {}
        act_2_state = act_2_block.get("state", {}) or {}
        act_3_block = meta.get("act_3", {}) or {}
        act_3_meta = act_3_block.get("metadata", {}) or {}
        act_3_state = act_3_block.get("state", {}) or {}
        act_4_block = meta.get("act_4", {}) or {}
        act_4_meta = act_4_block.get("metadata", {}) or {}
        act_4_state = act_4_block.get("state", {}) or {}
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
                user_profile={"age": "", "gender": ""},
                user_age="",
                user_gender="",
            )
            st.session_state.chat_history = []
            st.rerun()
    # Show only the LAST assistant message (current question) - BUT ONLY ON QUESTION PAGE
    if st.session_state.chat_history and st.session_state.ui_page == "question":
        # Find the last assistant message
        last_assistant_msg = None
        for t in reversed(st.session_state.chat_history):
            if t["role"] == "assistant":
                last_assistant_msg = t
                break

        if last_assistant_msg:
            with st.chat_message("assistant"):
                st.markdown(last_assistant_msg["content"], unsafe_allow_html=True)



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

    # Extract courses status
    courses_block = stage_meta.get("courses", {}) or {}
    courses_meta = courses_block.get("metadata", {}) or {}
    courses_turn = courses_meta.get("courses_turn", 0)

    course_selection_complete = st.session_state.state.get("course_selection_complete", False)
    # Check summary status
    summary_block = stage_meta.get("summary", {}) or {}
    summary_meta = summary_block.get("metadata", {}) or {}
    confirm_summary = summary_meta.get("confirm_summary", "unclear")
    summary_text = summary_meta.get("summary_text", "")
    recommendations = summary_meta.get("recommendations", [])
    summary_turn = summary_block.get("state", {}).get("turn", 0)  # ✅ FIX: Get turn from state, not metadata

    # Conversation is complete when all four acts are clear AND courses AND summary are done
    conversation_ended = (confirm_act_1 == "clear" and
                          confirm_act_2 == "clear" and
                          confirm_act_3 == "clear" and
                          confirm_act_4 == "clear" and
                          course_selection_complete and
                          summary_turn > 0)  # Summary has run

    # ========== SHOW BUTTONS OR COMPLETION MESSAGE ==========
    if conversation_ended:
        # Show completion message
        st.success(f"✅ **Conversation Complete!**")
        st.info(
            f"**Intent:** {act_1_type} | **Emotional Tone:** {act_2_emo_tone} | **Motivation:** {act_3_type} | **Barrier:** {act_4_type}")

        if summary_text:
            st.markdown("### 📋 Your Personalized Summary")
            st.markdown(summary_text)

        if recommendations:
            st.markdown("### 💡 Next Steps")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

        st.markdown("---")
        st.markdown("We've captured your preferences and will tailor your experience accordingly. 🎉")

    else:
        # ========== CHECK IF WE SHOULD SHOW COURSE SELECTION SCREENS ==========
        # Course selection happens AFTER Act 4 but BEFORE final completion
        act_4_block = stage_meta.get("act_4", {}) or {}
        act_4_meta = act_4_block.get("metadata", {}) or {}
        confirm_act_4 = act_4_meta.get("confirm_act_4", "unclear")

        # Check if act_4 is complete and course selection hasn't been completed
        if (confirm_act_4 == "clear" and
                not st.session_state.state.get("course_selection_complete", False)):

            # Initialize course selection if not started
            if "current_course_index" not in st.session_state:
                st.session_state.current_course_index = 0

            selected_courses = st.session_state.state.get("selected_courses", [])

            if selected_courses and st.session_state.current_course_index < len(selected_courses):
                # Show current course screen
                current_course = selected_courses[st.session_state.current_course_index]

                # Get question and image for this course
                question_text = COURSE_QUESTIONS.get(current_course, "Would you like to learn this?")
                image_path = COURSE_IMAGES.get(current_course, "")

                # Display the course screen
                st.markdown("---")
                st.markdown(
                    f"<div style='text-align: center; font-size: 28px; margin-bottom: 30px;'>{question_text}</div>",
                    unsafe_allow_html=True)

                # Display image if it exists
                if image_path and os.path.exists(image_path):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(image_path, use_container_width=True)
                else:
                    st.warning(f"Image not found: {image_path}")

                # Yes/No buttons
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Yes", key=f"course_yes_{st.session_state.current_course_index}",
                                 use_container_width=True, type="primary"):
                        # Record Yes response
                        course_responses = st.session_state.state.get("course_responses", {})
                        course_responses[current_course] = "Yes"
                        st.session_state.state = {**st.session_state.state, "course_responses": course_responses}

                        # Move to next course
                        st.session_state.current_course_index += 1
                        st.rerun()

                with col2:
                    if st.button("No", key=f"course_no_{st.session_state.current_course_index}",
                                 use_container_width=True):
                        # Record No response
                        course_responses = st.session_state.state.get("course_responses", {})
                        course_responses[current_course] = "No"
                        st.session_state.state = {**st.session_state.state, "course_responses": course_responses}

                        # Move to next course
                        st.session_state.current_course_index += 1
                        st.rerun()

                st.stop()  # Don't show anything else while in course selection




            else:
                # All courses completed - trigger summary
                st.session_state.state = {
                    **st.session_state.state,
                    "course_selection_complete": True
                }
                # Reset course index for next time
                if "current_course_index" in st.session_state:
                    st.session_state.current_course_index = 0
                # Trigger the graph to route to summary
                ai_text, new_state = process_user_message(st.session_state.graph, st.session_state.state, "")
                st.session_state.state = new_state
                st.rerun()


        # SPECIAL CASE: Hook completion - show Continue button
        hook_block = stage_meta.get("hook", {}) or {}
        hook_meta = hook_block.get("metadata", {}) or {}
        hook_status = hook_meta.get("hook_status", "unclear")
        last_agent = st.session_state.state.get("last_agent", "")

        if hook_status == "clear" and last_agent == "hook":
            if st.button("Continue", key="hook_continue", type="primary"):
                # Trigger graph to move to demographics
                ai_text, new_state = process_user_message(st.session_state.graph, st.session_state.state, "")
                st.session_state.state = new_state

                # Clear chat history - hook should not persist
                st.session_state.chat_history = []

                # Add demographics question to chat
                if ai_text:
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_text})



                st.rerun()
            st.stop()  # Don't show anything else

        # Check what UI page we're on
        if st.session_state.ui_page == "affirmation":
            # ============================================================
            # PAGE 2: AFFIRMATION PAGE
            # ============================================================
            st.markdown("---")
            # Display the matched affirmation - LARGE and CENTERED
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<h2 style='text-align: center; color: #4CAF50; font-size: 32px; line-height: 1.4;'>{st.session_state.current_matched_affirmation}</h2>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Display image (same as question page)
            try:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.image(st.session_state.current_image_path, width=300)
            except Exception as e:
                print(f"Could not load affirmation page image: {e}")

            # Continue button
            st.markdown("<br><br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue", key="continue_btn", type="primary", use_container_width=True):
                    # Check if we have a preloaded question ready
                    if st.session_state.next_question_preloaded and st.session_state.next_state_preloaded:
                        # Use preloaded question (instant!)
                        ai_text = st.session_state.next_question_preloaded
                        new_state = st.session_state.next_state_preloaded

                        # Update state
                        st.session_state.state = new_state

                        # Clear chat history and add new question
                        st.session_state.chat_history = []
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_text})

                        # Store affirmations from new state for next round
                        stage_meta_new = new_state.get("stage_meta", {}) or {}
                        last_agent_new = new_state.get("last_agent", "")
                        if last_agent_new in ["act_1", "act_2", "act_3", "act_4"]:
                            agent_block = stage_meta_new.get(last_agent_new, {}) or {}
                            agent_state = agent_block.get("state", {}) or {}
                            st.session_state.last_affirmations = agent_state.get("affirmations", [])

                        # Clear preload
                        st.session_state.next_question_preloaded = None
                        st.session_state.next_state_preloaded = None

                        # Switch back to question page
                        st.session_state.ui_page = "question"
                    else:
                        # No preload - just switch back to question page
                        st.session_state.ui_page = "question"

                    st.rerun()
        else:
            # ============================================================
            # PAGE 1: QUESTION PAGE
            # ============================================================

            # CRITICAL CHECK: Skip question form if Act 4 is complete and course selection is pending
            act_4_block = stage_meta.get("act_4", {}) or {}
            act_4_meta = act_4_block.get("metadata", {}) or {}
            confirm_act_4 = act_4_meta.get("confirm_act_4", "unclear")

            if (confirm_act_4 == "clear" and
                    not st.session_state.state.get("course_selection_complete", False)):
                # Act 4 complete, course selection pending - don't show form
                # The course selection UI block above will handle display
                st.stop()

            if (confirm_act_4 == "clear" and
                    not st.session_state.state.get("course_selection_complete", False)):
                # Act 4 complete, course selection pending - don't show form
                # The course selection UI block above will handle display
                st.stop()

            # Update image for current question
            last_agent = st.session_state.state.get("last_agent", "")


            if last_agent in ["act_1", "act_2", "act_3", "act_4"]:
                agent_block = stage_meta.get(last_agent, {}) or {}
                agent_state = agent_block.get("state", {}) or {}
                turn = agent_state.get("turn", 0)
                question_key = f"{last_agent}_turn_{turn}"
            else:
                question_key = last_agent

            # Only update image when question changes
            if question_key != st.session_state.current_question_key:
                st.session_state.current_question_key = question_key
                st.session_state.current_image_path = get_image_path_from_metadata(st.session_state.state)



            # Get current options from state
            options = st.session_state.state.get("last_options", []) or []



            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                # Determine response format from stage_meta
                last_agent = st.session_state.state.get("last_agent", "")

                # Get response format based on current agent
                response_format = "multiple_choice"  # default
                scale_range = ""
                scale_labels = {}

                if last_agent == "act_1":
                    act_1_block = stage_meta.get("act_1", {}) or {}
                    act_1_state = act_1_block.get("state", {}) or {}
                    act_1_turn = act_1_state.get("turn", 0)

                    # Get affirmations from the last conversation turn (most recent question)
                    conversation_history = st.session_state.state.get("conversation_history", [])
                    if conversation_history and len(conversation_history) > 0:
                        last_turn = conversation_history[-1]
                        st.session_state.last_affirmations = last_turn.get("affirmations", [])
                    else:
                        st.session_state.last_affirmations = act_1_state.get("affirmations", [])


                    if act_1_turn == 1:
                        response_format = act_1_state.get("response_format_1", "multiple_choice")
                    elif act_1_turn == 2:
                        response_format = act_1_state.get("response_format_2", "multiple_choice")
                    elif act_1_turn == 3:
                        response_format = act_1_state.get("response_format_3", "multiple_choice")
                    elif act_1_turn == 4:
                        response_format = act_1_state.get("response_format_4", "multiple_choice")
                    else:
                        response_format = "multiple_choice"

                    scale_range = act_1_state.get("scale_range", "")
                    scale_labels = act_1_state.get("scale_labels", {})

                elif last_agent == "act_2":
                    act_2_block = stage_meta.get("act_2", {}) or {}
                    act_2_state = act_2_block.get("state", {}) or {}
                    act_2_turn = act_2_state.get("turn", 0)

                    # Store affirmations for this question
                    st.session_state.last_affirmations = act_2_state.get("affirmations", [])

                    if act_2_turn == 1:
                        response_format = act_2_state.get("response_format_1", "multiple_choice")
                    elif act_2_turn == 2:
                        response_format = act_2_state.get("response_format_2", "multiple_choice")
                    elif act_2_turn == 3:
                        response_format = act_2_state.get("response_format_3", "multiple_choice")
                    elif act_2_turn == 4:
                        response_format = act_2_state.get("response_format_4", "multiple_choice")
                    elif act_2_turn == 5:
                        response_format = act_2_state.get("response_format_5", "multiple_choice")
                    elif act_2_turn == 6:
                        response_format = act_2_state.get("response_format_6", "multiple_choice")
                    else:
                        response_format = "multiple_choice"

                    scale_range = act_2_state.get("scale_range", "")
                    scale_labels = act_2_state.get("scale_labels", {})



                elif last_agent == "act_3":

                    act_3_block = stage_meta.get("act_3", {}) or {}

                    act_3_state = act_3_block.get("state", {}) or {}

                    st.session_state.last_affirmations = act_3_state.get("affirmations", [])

                    # ✅ FIX: read response_format from Act 3 state

                    response_format = act_3_state.get("response_format", "multiple_choice")

                    # ✅ FIX: pass scale metadata (same as Act 2)

                    scale_range = act_3_state.get("scale_range", "")

                    scale_labels = act_3_state.get("scale_labels", {})


                elif last_agent == "act_4":
                    act_4_block = stage_meta.get("act_4", {}) or {}
                    act_4_state = act_4_block.get("state", {}) or {}
                    st.session_state.last_affirmations = act_4_state.get("affirmations", [])
                    # ✅ FIX: Read response_format from Act 4 state (same as Act 3)
                    response_format = act_4_state.get("response_format", "multiple_choice")
                    # ✅ FIX: Pass scale metadata (same as Act 2 and Act 3)
                    scale_range = act_4_state.get("scale_range", "")
                    scale_labels = act_4_state.get("scale_labels", {})

                user_response = None

                # RENDER BASED ON RESPONSE FORMAT
                if response_format == "scale":
                    # SCALE INPUT
                    st.write("Rate your response:")

                    if scale_range and "-" in scale_range:
                        min_val, max_val = map(int, scale_range.split("-"))
                    else:
                        min_val, max_val = 1, 10

                    if scale_labels:
                        min_label = scale_labels.get("min", "")
                        max_label = scale_labels.get("max", "")
                        if min_label and max_label:
                            st.caption(f"**{min_val}:** {min_label} | **{max_val}:** {max_label}")

                    scale_value = st.slider(
                        "Select a value:",
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        key="scale_slider"
                    )

                    if st.button("Submit", key="scale_submit"):
                        user_response = str(scale_value)

                elif response_format == "image_select":
                    # IMAGE SELECT
                    st.markdown("<br><br>", unsafe_allow_html=True)

                    if options and len(options) > 0:
                        valid_images = [opt for opt in options if opt and opt.strip()]

                        if valid_images:
                            num_images = len(valid_images)
                            if num_images == 1:
                                cols = st.columns(1)
                            elif num_images == 2:
                                cols = st.columns(2)
                            else:
                                cols = st.columns(2)

                            for i, img_path in enumerate(valid_images[:4]):
                                col_idx = i % 2
                                with cols[col_idx]:
                                    try:
                                        st.image(img_path, width=300)
                                        filename = img_path.split('/')[-1].replace('.png', '').replace('.jpg',
                                                                                                       '').replace('_',
                                                                                                                   ' ').title()
                                        if st.button(filename, key=f"img_{i}", use_container_width=True):
                                            user_response = img_path
                                    except Exception as e:
                                        st.error(f"Could not load image: {img_path}")
                        else:
                            st.warning("No images available for selection.")
                    else:
                        st.warning("No images available for selection.")

                elif response_format == "yes_no":
                    # YES/NO INPUT
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    if len(options) >= 2:
                        if col1.button(options[0], key="yes_no_0", use_container_width=True):
                            user_response = options[0]
                        if col2.button(options[1], key="yes_no_1", use_container_width=True):
                            user_response = options[1]

                elif response_format == "either_or":
                    # EITHER/OR INPUT
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    if len(options) >= 2:
                        if col1.button(options[0], key="either_or_0", use_container_width=True):
                            user_response = options[0]
                        if col2.button(options[1], key="either_or_1", use_container_width=True):
                            user_response = options[1]

                else:
                    # MULTIPLE CHOICE
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if len(options) > 0:
                        cols = st.columns(len(options))
                        for i, opt in enumerate(options):
                            if cols[i].button(opt, key=f"opt_{i}", use_container_width=True):
                                user_response = opt

                # ========== PROCESS USER RESPONSE ==========
                if user_response:
                    # Get current agent to check if we should use 2-page flow
                    current_agent = st.session_state.state.get("last_agent", "")

                    # ===== CHECK IF WE NEED TO SHOW COURSES AFTER ACT 4 =====
                    stage_meta = st.session_state.state.get("stage_meta", {})
                    act_4_meta = stage_meta.get("act_4", {}).get("metadata", {})
                    act_4_state = stage_meta.get("act_4", {}).get("state", {})
                    act_4_turn = act_4_state.get("turn", 0)
                    courses_done = st.session_state.state.get("course_selection_complete", False)

                    # If this is Act 4 turn 4 (last question) and courses not done, redirect to courses
                    if current_agent == "act_4" and act_4_turn == 4 and not courses_done:
                        # Process the final Act 4 answer first
                        ai_text, new_state = process_user_message(
                            st.session_state.graph,
                            st.session_state.state,
                            user_response
                        )
                        st.session_state.state = new_state

                        # Now redirect to course selection instead of continuing
                        st.session_state.chat_history = []
                        st.rerun()
                        return  # Exit here to show courses

                    # ===== FIX: Handle Act 3 SCALE answers exactly like Act 2 =====
                    if current_agent == "act_3":
                        stage_meta = st.session_state.state.get("stage_meta", {})
                        act_3_state = stage_meta.get("act_3", {}).get("state", {})

                        if act_3_state.get("response_format") == "scale":
                            try:
                                numeric_value = int(user_response)
                            except (TypeError, ValueError):
                                numeric_value = None

                            if numeric_value is not None:
                                scale_mapping = act_3_state.get("scale_mapping", {})

                                semantic_value = numeric_value  # fallback

                                for key, value in scale_mapping.items():
                                    if "-" in key:
                                        lo, hi = map(int, key.split("-"))
                                        if lo <= numeric_value <= hi:
                                            semantic_value = value
                                            break
                                    elif str(numeric_value) == key:
                                        semantic_value = value
                                        break

                                turn = act_3_state.get("turn", 1)
                                act_3_state[f"act_3_answer_{turn}"] = semantic_value

                                stage_meta["act_3"]["state"] = act_3_state
                                st.session_state.state["stage_meta"] = stage_meta

                    # NOT for demographics or hook
                    if current_agent in ["act_1", "act_2", "act_3", "act_4"]:
                        # CRITICAL: 2-page magic happens here

                        # Step 1: Match affirmation to user's answer
                        current_affirmations = st.session_state.last_affirmations
                        matched_affirmation = match_affirmation_to_answer(
                            user_response,
                            options,
                            current_affirmations
                        )
                        # Update the last conversation turn with matched affirmation and user's answer
                        if st.session_state.state.get("conversation_history"):
                            conversation_history = st.session_state.state["conversation_history"]
                            if len(conversation_history) > 0:
                                last_turn = conversation_history[-1]
                                last_turn["affirmation"] = matched_affirmation
                                last_turn["answer"] = user_response
                                # Update state
                                st.session_state.state["conversation_history"] = conversation_history

                    else:
                        # For demographics, hook, and other agents: use normal flow (no 2-page)
                        current_agent = st.session_state.state.get("last_agent", "")

                        # Process the demographics answer first
                        ai_text, new_state = process_user_message(
                            st.session_state.graph,
                            st.session_state.state,
                            user_response
                        )
                        st.session_state.state = new_state

                        # Check if demographics just completed
                        demo_block = new_state.get("stage_meta", {}).get("demographics", {})
                        demo_status = demo_block.get("metadata", {}).get("demo_status", "")

                        if current_agent == "demographics" and demo_status == "complete":
                            # Demographics just finished! Preload act_1 Q1 NOW
                            preload_text, preload_state = process_user_message(
                                st.session_state.graph,
                                new_state,
                                ""
                            )
                            st.session_state.state = preload_state
                            ai_text = preload_text  # Use the preloaded question immediately

                        st.session_state.chat_history = []
                        if ai_text:
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": ai_text})
                        st.rerun()

                    st.session_state.current_matched_affirmation = matched_affirmation

                    # Step 2: Immediately call LLM for next question
                    try:
                        ai_text, new_state = process_user_message(
                            st.session_state.graph,
                            st.session_state.state,
                            user_response
                        )
                        # Store for instant display when user clicks Continue
                        st.session_state.next_question_preloaded = ai_text
                        st.session_state.next_state_preloaded = new_state
                    except Exception as e:
                        print(f"Error preloading next question: {e}")
                        st.session_state.next_question_preloaded = None
                        st.session_state.next_state_preloaded = None

                    # Step 3: Switch to affirmation page
                    st.session_state.ui_page = "affirmation"
                    st.rerun()


                    # NOT for demographics or hook
                    if current_agent in ["act_1", "act_2", "act_3", "act_4"]:
                        # CRITICAL: 2-page magic happens here

                        # Step 1: Match affirmation to user's answer
                        current_affirmations = st.session_state.last_affirmations
                        matched_affirmation = match_affirmation_to_answer(
                            user_response,
                            options,
                            current_affirmations
                        )
                        # Update the last conversation turn with matched affirmation and user's answer
                        if st.session_state.state.get("conversation_history"):
                            conversation_history = st.session_state.state["conversation_history"]
                            if len(conversation_history) > 0:
                                last_turn = conversation_history[-1]
                                last_turn["affirmation"] = matched_affirmation
                                last_turn["answer"] = user_response
                                # Update state
                                st.session_state.state["conversation_history"] = conversation_history

                    else:
                        # For demographics, hook, and other agents: use normal flow (no 2-page)
                        current_agent = st.session_state.state.get("last_agent", "")

                        # Process the demographics answer first
                        ai_text, new_state = process_user_message(
                            st.session_state.graph,
                            st.session_state.state,
                            user_response
                        )
                        st.session_state.state = new_state

                        # Check if demographics just completed
                        demo_block = new_state.get("stage_meta", {}).get("demographics", {})
                        demo_status = demo_block.get("metadata", {}).get("demo_status", "")

                        if current_agent == "demographics" and demo_status == "complete":
                            # Demographics just finished! Preload act_1 Q1 NOW
                            preload_text, preload_state = process_user_message(
                                st.session_state.graph,
                                new_state,
                                ""
                            )
                            st.session_state.state = preload_state
                            ai_text = preload_text  # Use the preloaded question immediately

                        st.session_state.chat_history = []
                        if ai_text:
                            st.session_state.chat_history.append({"role": "assistant", "content": ai_text})
                        st.rerun()

                    st.session_state.current_matched_affirmation = matched_affirmation

                    # Step 2: Immediately call LLM for next question
                    try:
                        ai_text, new_state = process_user_message(
                            st.session_state.graph,
                            st.session_state.state,
                            user_response
                        )
                        # Store for instant display when user clicks Continue
                        st.session_state.next_question_preloaded = ai_text
                        st.session_state.next_state_preloaded = new_state
                    except Exception as e:
                        print(f"Error preloading next question: {e}")
                        st.session_state.next_question_preloaded = None
                        st.session_state.next_state_preloaded = None

                    # Step 3: Switch to affirmation page
                    st.session_state.ui_page = "affirmation"
                    st.rerun()


if __name__ == "__main__":
    main()