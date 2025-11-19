# app_enhanced_refactored.py – Sparky with inheritance-based agent architecture
from __future__ import annotations

import os, json, re, yaml
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Any, Tuple, Sequence, Annotated
from abc import ABC, abstractmethod

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

DEFAULT_CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", "new_config.yaml")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY and ChatOpenAI)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.8"))

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

from langchain_core.pydantic_v1 import BaseModel as PydanticV1BaseModel, Field as PydanticV1Field


# ============================================================================
# AD DATA INTEGRATION
# ============================================================================

class AdData(TypedDict):
    """External ad data that influences the conversation"""
    ad_id: str
    campaign_name: str
    ad_creative_theme: str  # e.g., "creativity", "wellness", "side_hustle"
    target_audience: str  # e.g., "busy_professionals", "creative_explorers"
    landing_page_url: str
    utm_source: str
    utm_campaign: str
    timestamp: str


def get_mock_ad_data() -> AdData:
    """Mock ad data - replace this with actual external data source"""
    return {
        "ad_id": "AD_12345",
        "campaign_name": "Creative Spark 2024",
        "ad_creative_theme": "creativity",
        "target_audience": "creative_explorers",
        "landing_page_url": "https://nowyouknow.com/spark",
        "utm_source": "facebook",
        "utm_campaign": "creative_spark_nov",
        "timestamp": "2024-11-18T10:30:00Z",
    }


def get_ad_data_from_external_source(ad_id: Optional[str] = None) -> AdData:
    """Fetch ad data from external source (API, database, etc.).
    For now returns mock data, but structured to be easily replaced."""
    # TODO: Replace with actual external data fetching
    return get_mock_ad_data()


# ============================================================================
# STRUCTURED OUTPUTS
# ============================================================================

class AgentResponse(PydanticV1BaseModel):
    """Generic structured response used by ALL agents.

    - question_text: conversational question or prompt.
    - options: list of answer choices.
    - metadata: agent-specific output (intent, motivation_type, barriers, persona, etc.).
    - state: agent-specific state (turn counts, levels, flags, progress).
    """

    question_text: str = PydanticV1Field(
        description="The conversational question or prompt to show the user"
    )

    options: List[str] = PydanticV1Field(
        description="Answer options (any number, typically 4)"
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

    # Per-agent metadata/state from last turns.
    # After the refactor, each agent will store:
    #   stage_meta[info_type] = {
    #       "metadata": {...},  # AgentResponse.metadata
    #       "state": {...},     # AgentResponse.state
    #   }
    stage_meta: Annotated[Dict[str, Dict[str, Any]], merge_stage_meta]

    # Last answer options shown to the user (for the UI to render)
    last_options: List[str]

    # Ad / acquisition context (where the user came from, campaign, theme, etc.)
    ad_data: AdData

    # Persistent, cross-agent user profile.
    user_profile: Dict[str, Any]


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

    def build_context(self, state: AgentState, user_text: str) -> Dict[str, str]:
        """Build the context dictionary for template rendering."""
        stage_meta = state.get("stage_meta", {}) or {}
        block = stage_meta.get(self.info_type, {}) or {}
        meta = block.get("metadata", {}) or {}
        last_state_json = block.get("state", {}) or {}

        turns_for_info = len(state.get("collected_info", {}).get(self.info_type, []))
        last_intent_status = (
            meta.get("confirm_intent")
            or meta.get("intent_status")
            or ""
        )

        # Build ad context
        ad_data = state.get("ad_data", {}) or {}
        ad_context = f"""Ad Context:
- Campaign: {ad_data.get('campaign_name', 'unknown')}
- Theme: {ad_data.get('ad_creative_theme', 'unknown')}
- Target Audience: {ad_data.get('target_audience', 'unknown')}
- Source: {ad_data.get('utm_source', 'unknown')}
"""

        # Build conversation history
        last_msgs = list(state.get("messages", []))[-8:]
        history = "\n".join([getattr(m, "content", "") for m in last_msgs])

        return {
            "history": history,
            "user_text": user_text,
            "last_agent": state.get("last_agent", "conversation"),
            "turns_for_info": str(turns_for_info),
            "last_intent": last_intent_status,
            "collected_info": json.dumps(state.get("collected_info", {}), ensure_ascii=False),
            "state_json": json.dumps(last_state_json, ensure_ascii=False),
            "ad_context": ad_context,
        }

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
            structured_llm = llm.with_structured_output(AgentResponse)
            response: AgentResponse = structured_llm.invoke(msgs)
            return response

        except Exception as e:
            print(f"Error with structured output: {e}")
            import traceback

            traceback.print_exc()
            mock_text = self.get_mock_response(user_text, state)
            return AgentResponse(
                question_text=mock_text,
                options=[],
                metadata={"error": str(e)},
                state={},
            )

    def process(self, state: AgentState) -> AgentState:
        """Main processing method called by LangGraph."""
        user_text = state.get("user_input", "")

        # Call the LLM (or mock) to get a structured AgentResponse
        response = self.generate_response(user_text, state)
        print(f"\n\n====== {self.name} AgentResponse ======\n", response, "\n===========================\n\n")

        # Build user-facing text: question + enumerated options
        display_text = response.question_text or ""
        if response.options:
            lines = []
            for i, opt in enumerate(response.options):
                label = chr(65 + i)  # A, B, C, ...
                lines.append(f"{label}) {opt}")
            display_text = display_text.rstrip() + "\n\n" + "\n".join(lines)

        # Update stage_meta for this agent with the new metadata and state
        import copy

        new_stage_meta = copy.deepcopy(state.get("stage_meta", {}))
        new_stage_meta[self.info_type] = {
            "metadata": dict(response.metadata or {}),
            "state": dict(response.state or {}),
        }

        print(f"DEBUG - {self.name} - metadata: {response.metadata}")
        print(f"DEBUG - {self.name} - state: {response.state}")

        # Update collected info
        merged_collected = dict(state.get("collected_info", {}))
        if user_text:
            merged_collected.setdefault(self.info_type, []).append(user_text)

        return AgentState(
            messages=[AIMessage(content=display_text)],
            collected_info=merged_collected,
            stage_meta=new_stage_meta,
            exchanges_with_current=state.get("exchanges_with_current", 0) + 1,
            last_agent=self.info_type,
        )

class SupervisorDecision(PydanticV1BaseModel):
    next_agent: str = PydanticV1Field(
        description="Which agent should run next. One of: connection, wellness, creativity, monetization, FINISH."
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
        "connection": ConnectionAgent,
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

        conn_block = stage_meta.get("connection", {}) or {}
        conn_meta = conn_block.get("metadata", {}) or {}

        conn_turns = len(collected.get("connection", []))

        intent_status = (
            conn_meta.get("confirm_intent")
            or conn_meta.get("intent_status")
            or ""
        ).lower()

        emotional_tone = (
            conn_meta.get("emo_tone")
            or conn_meta.get("emotional_tone")
            or ""
        ).lower()

        intent_type = (conn_meta.get("intent_type") or "").lower()

        ad_data = state.get("ad_data", {}) or {}
        ad_theme = ad_data.get("ad_creative_theme", "").lower()

        print(
            f"DEBUG SUPERVISOR (fallback) - conn_turns: {conn_turns}, "
            f"intent_status: {intent_status}, emotional_tone: {emotional_tone}, "
            f"ad_theme: {ad_theme}, intent_type: {intent_type}"
        )

        # Stage 1 exit readiness
        ready = (intent_status == "clear") and (emotional_tone != "resistant")
        if ready or conn_turns >= 3:
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
                f"(ready={ready}, conn_turns={conn_turns})"
            )
            return {"next_agent": "FINISH", "exchanges_with_current": 0}

        # Otherwise, stay in connection
        if "connection" in self.agent_keys:
            print("DEBUG SUPERVISOR (fallback) - Continuing with connection agent")
            return {
                "agent_index": self.agent_keys.index("connection"),
                "next_agent": "connection",
                "exchanges_with_current": 0,
            }

        # No connection agent present → finish
        return {"next_agent": "FINISH", "exchanges_with_current": 0}

    # ------------------------------------------------------------------
    # NEW: LLM-BASED SUPERVISOR CONTEXT BUILDING
    # ------------------------------------------------------------------
    def build_context(self, state: AgentState) -> Dict[str, str]:
        """
        Build the JSON-stringified context passed into the supervisor prompts.
        This is what the supervisor LLM will see.
        """
        return {
            "stage_meta": json.dumps(state.get("stage_meta", {}), ensure_ascii=False),
            "collected_info": json.dumps(state.get("collected_info", {}), ensure_ascii=False),
            "ad_data": json.dumps(state.get("ad_data", {}), ensure_ascii=False),
            "user_profile": json.dumps(state.get("user_profile", {}), ensure_ascii=False),
            "last_agent": state.get("last_agent", "") or "",
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
    # ------------------------------------------------------------------
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

    def route_from_supervisor(state: AgentState) -> str:
        nxt = (state.get("next_agent") or "").lower()
        if nxt in ("finish", "end"):
            return "END"
        return nxt if nxt in main_keys else (main_keys[0] if main_keys else "END")

    edges_map = {k: k for k in main_keys}
    edges_map.update({"END": END})
    workflow.add_conditional_edges("supervisor", route_from_supervisor, edges_map)

    for k in main_keys:
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
    if (state.get("next_agent") or "").upper() == "FINISH":
        state["last_options"] = []
        return "We've collected everything we need. Thanks!", state

    print(f"DEBUG - Input state stage_meta: {state.get('stage_meta', {})}")

    before = len(state.get("messages", []))
    state = {
        **state,
        "messages": list(state.get("messages", [])) + [HumanMessage(content=msg)],
        "user_input": msg,
    }
    new_state = graph.invoke(state)

    print(f"DEBUG - Output state stage_meta: {new_state.get('stage_meta', {})}")

    after = new_state.get("messages", [])
    ai_msgs = [m for m in after[before:] if isinstance(m, AIMessage)]
    raw_ai_text = "\n\n".join([m.content for m in ai_msgs if m.content])

    clean_text, options = extract_options(raw_ai_text)
    new_state["last_options"] = options

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
        ad_data = st.session_state.ad_data
        st.markdown(f"**Campaign:** {ad_data.get('campaign_name', 'N/A')}")
        st.markdown(f"**Theme:** {ad_data.get('ad_creative_theme', 'N/A')}")
        st.markdown(f"**Audience:** {ad_data.get('target_audience', 'N/A')}")
        st.markdown(f"**Source:** {ad_data.get('utm_source', 'N/A')}")

        st.markdown("---")
        st.header("Collected Info")
        for k, v in st.session_state.state.get("collected_info", {}).items():
            st.markdown(f"**{k}** ({len(v)})")
            for i, it in enumerate(v[-5:]):
                st.write(f"- {i + 1}. {it}")

        st.markdown("---")
        st.header("Conversation Insights")
        meta = st.session_state.state.get("stage_meta", {}) or {}
        conn_block = meta.get("connection", {}) or {}
        conn_meta = conn_block.get("metadata", {}) or {}
        conn_state = conn_block.get("state", {}) or {}

        intent_status = conn_meta.get("confirm_intent") or conn_meta.get("intent_status", "—")
        intent_type = conn_meta.get("intent_type", "—")
        emotional_tone = conn_meta.get("emo_tone") or conn_meta.get("emotional_tone", "—")
        behavioral_signal = conn_meta.get("behavioral_signal", "—")

        st.markdown(f"- **Confirm Intent:** {intent_status or '—'}")
        st.markdown(f"- **Intent Type:** {intent_type or '—'}")
        st.markdown(f"- **Emotional Tone:** {emotional_tone or '—'}")
        st.markdown(f"- **Behavioral Signal:** {behavioral_signal or '—'}")
        st.markdown(f"- **STATE:** {json.dumps(conn_state, ensure_ascii=False)}")

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

    # Buttons for current options (if any)
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
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_text})
                    st.rerun()

    # Free-text input
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
