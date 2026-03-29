# A/B Test Causal Contamination Forensics — OpenEnv Blueprint

> **Production-grade implementation guide for the OpenEnv hackathon submission.**  
> Deadline: April 1, 2025 | Framework: OpenEnv | Deployment: Hugging Face Spaces

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Data Models](#3-data-models)
4. [Action Space Design](#4-action-space-design)
5. [Observation Design](#5-observation-design)
6. [Environment Dynamics](#6-environment-dynamics)
7. [Task Design (Easy → Hard)](#7-task-design-easy--hard)
8. [Reward Function](#8-reward-function)
9. [Grader Design](#9-grader-design)
10. [Baseline Agent Design](#10-baseline-agent-design)
11. [Inference Script (inference.py)](#11-inference-script-inferencepy)
12. [OpenEnv Compliance](#12-openenv-compliance)
13. [Project Structure](#13-project-structure)
14. [Docker + Deployment](#14-docker--deployment)
15. [Development Phase Plan](#15-development-phase-plan)
16. [Failure Modes & Debugging Guide](#16-failure-modes--debugging-guide)
17. [Evaluation Strategy](#17-evaluation-strategy)

---

## 1. System Overview

### Problem Explanation (Deep Technical Framing)

Every product experimentation team at scale — Meta, Google, Airbnb, Shopify — faces a category of error more dangerous than a failed experiment: a *contaminated* experiment that *looks* valid. When an experiment result is invalid, the worst outcome is not rejecting a good idea. The worst outcome is **shipping a feature that appears effective but isn't**, leading to mis-attributed revenue impact, incorrect product roadmap decisions, and corrupted future experiments that inherit the flawed baseline.

This environment simulates an **experimentation auditing platform** where:
- Experiments generate aggregate statistical results with surface-level significance.
- Some results are genuinely valid; others are contaminated by hidden statistical pathologies.
- The agent's job is **not** to maximize a metric — it is to determine whether the result can be trusted, identify the contamination mechanism if present, and produce a structured invalidation report.

This is a **reasoning under uncertainty with progressive information disclosure** task — the agent must decide *what to investigate*, in *what order*, with a *finite action budget*.

### Why This Environment Is Novel (Benchmark Gap)

| Existing Benchmarks | This Environment |
|---|---|
| Agent analyzes data → produces insight | Agent audits process → produces validity verdict |
| Observation is complete at episode start | Observation is progressively unlocked by agent actions |
| Correct answer is a metric or label | Correct answer is a structured multi-component report |
| No calibration scoring | Confidence calibration is a scored dimension |
| No false-positive penalty | Incorrectly flagging a clean result is penalized |
| Single contamination mechanism | Up to 3 simultaneous contamination layers (Task 3) |

No existing OpenEnv environment targets **statistical validity reasoning** — the skill that separates a junior analyst from a senior experimentation scientist.

### Real-World Mapping

This environment directly models the **internal experimentation review pipeline** at large technology companies:

- **Meta's XP system** flags hundreds of potentially contaminated experiments per month for human review.
- **Google's CUPED/SUTVA tooling** partially automates SRM checks but still relies on analyst judgment for complex interference cases.
- **Airbnb's experimentation platform** documented Simpson's paradox cases that nearly caused incorrect product launches.
- The **senior data scientist role** responsible for experiment validity review is a scarce, high-cost position. An agent capable of performing this audit would have immediate economic value.

### Agent Learning Objective

The agent must learn:
1. **What to investigate** — which queries are informative for which contamination types.
2. **When to stop** — balancing investigation depth against action budget.
3. **How to synthesize** — combining evidence from multiple queries into a coherent verdict.
4. **How to calibrate** — expressing uncertainty proportional to evidence strength.
5. **False-positive avoidance** — resisting the urge to flag a clean experiment just because something looks suspicious.

The agent is **NOT** learning to maximize a business metric. It is learning the meta-skill of experimental validity reasoning.

---

## 2. Architecture Design

### 2.1 Environment Core

**Responsibilities:**
- Orchestrate the episode lifecycle (reset → step → terminate).
- Maintain the hidden `ContaminationSpec` for the current episode.
- Route actions to the appropriate sub-system.
- Accumulate the episode log for grading.

**Data Flow:**
```
reset() → TaskGenerator.sample() → StateManager.init() → ObservationBuilder.build_initial() → Observation

step(action) → ActionExecutor.validate(action)
             → StateManager.update(action)
             → RewardEngine.compute(action, state, spec)
             → ObservationBuilder.build_updated(state)
             → (Observation, Reward, done, info)
```

**Key Design Decisions:**
- The `ContaminationSpec` is created once per episode and never exposed to the agent through any observation field or info dict.
- The `done` flag triggers on: (a) terminal action (`flag_contamination`, `approve_result`, `request_rerun`), (b) action budget exhausted (max 15 steps), or (c) invalid action submitted 3 times consecutively.
- Episode random seed is set from `task_id + episode_seed` to ensure reproducibility.

**Edge Cases:**
- Agent submits the same query action twice: second call returns cached result (no new information, small penalty applied).
- Agent submits a `flag_contamination` action with a missing required parameter: action is rejected, counts against invalid action budget.
- Agent submits `approve_result` on a contaminated episode: episode ends immediately with maximum negative terminal reward.

---

### 2.2 State Manager

**Responsibilities:**
- Maintain all mutable episode state.
- Track which queries have been executed.
- Maintain the progressive information unlock map.
- Enforce action budget.

**State Structure:**
```python
@dataclass
class EpisodeState:
    experiment_id: str
    step_count: int
    max_steps: int
    executed_queries: List[str]           # action types that have been run
    revealed_data: Dict[str, Any]         # populated as queries are executed
    invalid_action_count: int
    episode_done: bool
    termination_reason: Optional[str]
    episode_log: List[Dict]               # full action history with timestamps
```

**Key Design Decisions:**
- `revealed_data` is append-only. Once a data type is revealed, it persists in all subsequent observations.
- Step count is visible to the agent in the observation (budget awareness).
- The same query type cannot be submitted twice — second attempt returns `{"cached": true, "data": <previous_result>}` and costs 0.03 reward.

---

### 2.3 Observation Builder

**Responsibilities:**
- Compose the `Observation` Pydantic model from current state.
- Apply information unlock logic — what is visible vs. hidden.
- Generate realistic synthetic data consistent with the `ContaminationSpec`.

**Data Flow:**
```
ContaminationSpec → DataGenerator → raw_data_tables
StateManager.revealed_data → filter(raw_data_tables) → visible_fields
visible_fields → ObservationBuilder → Observation Pydantic model
```

**Key Design Decisions:**
- The `DataGenerator` uses the `ContaminationSpec` to produce internally consistent data. For example, if the spec defines `contamination_type="srm"`, the generated `aggregate_results.control_count` and `treatment_count` will reflect the mismatch.
- Noise is added deterministically using the episode seed — same seed always produces same data.
- The `available_queries` field always lists all possible action types, regardless of what has been executed. This prevents the agent from inferring information from what's *not* listed.

**Progressive Unlock Logic:**
```
Initial state:   experiment_id, hypothesis, primary_metric, aggregate_results, 
                 experiment_metadata, available_queries, steps_remaining

After run_srm_check:          + randomization_check (SRM p-value, expected vs actual split)
After query_subgroup(dim):    + subgroup_results[dim] (per-dimension breakdown)
After query_temporal:         + temporal_breakdown (day-by-day metric values)
After query_assignment_overlap: + user_assignment_overlap (cross-experiment matrix)
After check_network_exposure: + network_exposure_map (spillover percentages)
After query_secondary_metrics: + secondary_metric_results (guardrail metrics)
After compute_mde:            + mde_analysis (required vs actual sample sizes)
After inspect_randomization:  + randomization_audit (algorithm, seed, assignment log)
```

---

### 2.4 Action Executor

**Responsibilities:**
- Validate incoming actions against schema.
- Check action preconditions (e.g., budget remaining).
- Trigger the appropriate data reveal in StateManager.
- Return structured error messages for invalid actions.

**Key Design Decisions:**
- Actions are validated with Pydantic before execution. Malformed JSON → `ActionValidationError` with specific field-level error messages.
- The `reasoning` field in every action is stored in the episode log and used by the grader for partial credit scoring on evidence quality.
- The `confidence` field (0.0–1.0) is required for terminal actions and optional for investigative actions.

---

### 2.5 Reward Engine

**Responsibilities:**
- Compute step-level rewards for investigative actions.
- Compute terminal rewards for verdict actions.
- Apply efficiency penalties.
- Apply calibration rewards.

**Key Design Decisions:**
- Reward computation requires access to the hidden `ContaminationSpec`. This is the only component (besides the Grader) with access to ground truth.
- Rewards are accumulated in the episode log for transparency in debugging.
- The calibration reward uses a continuous scoring function, not a threshold — partial credit for approximate confidence calibration.

---

### 2.6 Task Generator (ContaminationSpec Generator)

**Responsibilities:**
- Sample a `ContaminationSpec` for each episode.
- Generate internally consistent synthetic experiment data matching the spec.
- Ensure each task has deterministic spec when seeded.

**Key Design Decisions:**
- Each task (`task_id = 1, 2, 3, 4`) has a fixed set of possible `ContaminationSpec` instances (3–5 variants per task). This ensures difficulty consistency across eval runs.
- The DataGenerator produces plausible real-world numbers: sample sizes between 10K–500K, effect sizes between -5% and +15%, p-values computed from the synthetic data rather than fabricated.
- Task 4 (clean experiment) is generated to have surface-level suspicious features that resolve to innocent explanations — this requires deliberate inverse generation logic.

---

### 2.7 Grader System

**Responsibilities:**
- Evaluate the complete episode log against the hidden `ContaminationSpec`.
- Produce a structured score breakdown across 5 dimensions.
- Output a final scalar score in [0.0, 1.0].

**Design Principle:** Zero LLM-as-judge. All scoring is computed from structured data comparisons.

---

### 2.8 API Layer (OpenEnv Interface)

**Responsibilities:**
- Expose `step()`, `reset()`, `state()` HTTP endpoints.
- Serialize/deserialize Pydantic models.
- Handle concurrent sessions (session ID–based isolation).
- Return proper HTTP status codes and error formats.

**Endpoints:**
```
POST /reset         Body: {task_id: int, seed: int}  → Observation
POST /step          Body: AuditAction                → StepResult
GET  /state         Query: ?session_id=...           → EpisodeState
GET  /tasks         → List[TaskDefinition]
GET  /health        → {status: "ok"}
```

---

### 2.9 Logging & Episode Tracking

- Every action, observation delta, and reward is logged to a structured JSON episode log.
- Episode logs are written to `/logs/episodes/{session_id}.jsonl`.
- The baseline inference script reads these logs to produce reproducible scores.

---

## 3. Data Models

### 3.1 Observation

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal
from datetime import date

class AggregateResult(BaseModel):
    control_mean: float            # e.g., 0.412 (41.2% retention)
    treatment_mean: float          # e.g., 0.433 (43.3% retention)
    relative_lift: float           # e.g., 0.051 (5.1% lift)
    absolute_lift: float           # e.g., 0.021
    p_value: float                 # e.g., 0.001
    control_count: int             # e.g., 61847
    treatment_count: int           # e.g., 48203
    confidence_interval_lower: float
    confidence_interval_upper: float

class ExperimentMeta(BaseModel):
    start_date: date               # e.g., 2024-01-15
    end_date: date                 # e.g., 2024-02-15
    targeting_rule: str            # e.g., "all_users_18+ in US"
    intended_split: float          # e.g., 0.50 (50% treatment)
    randomization_unit: str        # e.g., "user_id"
    platform: str                  # e.g., "mobile_ios"
    experiment_owner: str          # e.g., "growth_team"
    hypothesis: str

class DailyResult(BaseModel):
    date: date
    control_mean: float
    treatment_mean: float
    relative_lift: float
    control_count: int
    treatment_count: int

class SubgroupResult(BaseModel):
    dimension: str                 # e.g., "device_type"
    value: str                     # e.g., "mobile"
    control_mean: float
    treatment_mean: float
    relative_lift: float
    control_count: int
    treatment_count: int

class SRM_Result(BaseModel):
    expected_split: float          # e.g., 0.50
    actual_split: float            # e.g., 0.438
    chi_square_statistic: float    # e.g., 1847.3
    p_value: float                 # e.g., 0.0000001
    srm_detected: bool             # True if p < 0.001
    severity: Literal["none", "mild", "severe"]

class OverlapMatrix(BaseModel):
    experiment_ids: List[str]
    overlap_fractions: Dict[str, Dict[str, float]]  # [exp_a][arm] → fraction also in [exp_b][arm]
    # Example: {"exp_X": {"control": {"exp_Y_treatment": 0.72}}}

class MDEAnalysis(BaseModel):
    observed_effect_size: float
    required_sample_per_arm: int   # to detect observed_effect_size at 80% power
    actual_sample_per_arm: int
    achieved_power: float          # actual power given sample size
    underpowered: bool

class ExperimentObservation(BaseModel):
    """The full observation returned to the agent."""
    session_id: str
    experiment_id: str
    primary_metric: str
    aggregate_results: AggregateResult
    experiment_metadata: ExperimentMeta
    available_queries: List[str]   # always full list
    steps_taken: int
    steps_remaining: int

    # Populated only after agent queries them
    subgroup_results: Optional[Dict[str, List[SubgroupResult]]] = None
    temporal_breakdown: Optional[List[DailyResult]] = None
    user_assignment_overlap: Optional[OverlapMatrix] = None
    randomization_check: Optional[SRM_Result] = None
    network_exposure_map: Optional[Dict[str, float]] = None
    secondary_metric_results: Optional[Dict[str, AggregateResult]] = None
    mde_analysis: Optional[MDEAnalysis] = None
    randomization_audit: Optional[Dict] = None
    peer_experiment_list: Optional[List[Dict]] = None
```

---

### 3.2 Action

```python
class AuditAction(BaseModel):
    action_type: Literal[
        "query_subgroup",
        "query_temporal",
        "run_srm_check",
        "query_assignment_overlap",
        "check_network_exposure",
        "inspect_randomization",
        "query_secondary_metrics",
        "compute_mde",
        "flag_contamination",
        "approve_result",
        "request_rerun"
    ]
    parameters: Dict = Field(default_factory=dict)
    reasoning: str = Field(..., min_length=10, max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # For flag_contamination, parameters must include:
    # {
    #   "contamination_type": str,             # one of the ContaminationSpec types
    #   "evidence_facts": List[str],           # specific factual claims from observed data
    #   "estimated_true_effect": float,        # agent's estimate of real effect (optional)
    #   "recommended_action": str              # "rerun" | "discard" | "partial_rerun"
    # }

    # For query_subgroup, parameters must include:
    # {"dimension": str}   # e.g., "device_type", "country", "enrollment_cohort"
```

---

### 3.3 Reward

```python
class StepReward(BaseModel):
    step_reward: float
    components: Dict[str, float]   # breakdown by component
    cumulative_reward: float
    reasoning: str                 # human-readable explanation

class EpisodeReward(BaseModel):
    total_reward: float
    step_rewards: List[StepReward]
    terminal_reward: float
    efficiency_penalty: float
    calibration_reward: float
```

---

### 3.4 ContaminationSpec (Hidden State)

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ContaminationSpec:
    contamination_type: str        # one of the 8 types listed below
    true_effect_size: float        # actual causal effect (0.0 for most contaminated)
    visible_effect_size: float     # what aggregate metrics show
    optimal_investigation_steps: int  # min steps needed to reach correct verdict

    # Type-specific fields
    contaminated_subgroup: Optional[str] = None      # Simpson's paradox
    interference_experiment_id: Optional[str] = None  # SUTVA violation
    novelty_half_life_days: Optional[int] = None      # novelty effect
    srm_actual_split: Optional[float] = None          # SRM
    network_spillover_fraction: Optional[float] = None
    required_queries: Optional[List[str]] = None      # queries needed for full credit
    ground_truth_evidence: Optional[Dict] = None      # facts the grader checks against

# Contamination types:
# "clean"                  — result is valid, should be approved
# "srm"                    — sample ratio mismatch (randomization bug)
# "sutva_violation"        — inter-experiment interference
# "novelty_effect"         — temporal decay of treatment effect
# "simpsons_paradox"       — aggregate masks subgroup reversal
# "network_spillover"      — treatment leaked to control via social graph
# "multiple_testing"       — p-hacked across 15 metrics
# "underpowered_overclaim" — MDE >> claimed effect, result is noise
```

---

## 4. Action Space Design

### 4.1 `query_subgroup`

- **Purpose:** Break aggregate results down by a specified dimension to detect compositional imbalances (Simpson's paradox).
- **Parameters:** `{"dimension": "device_type" | "country" | "enrollment_cohort" | "user_segment" | "platform_version"}`
- **State change:** Populates `observation.subgroup_results[dimension]` with per-group `SubgroupResult` objects.
- **When to use:** When aggregate results look unusually clean, or when the targeting rule suggests non-uniform group composition.
- **Failure modes:** Agent queries a dimension that is not informative for the current contamination type (penalty: -0.03). Agent queries the same dimension twice (cached result, small penalty).

### 4.2 `query_temporal`

- **Purpose:** Get day-by-day metric breakdown to detect novelty effects or temporal confounds.
- **Parameters:** `{}` (no parameters; returns full temporal breakdown)
- **State change:** Populates `observation.temporal_breakdown` with `List[DailyResult]`.
- **When to use:** When the experiment ran for >7 days and results claim long-term significance.
- **Failure modes:** Agent ignores this action on a novelty effect episode because aggregate results look strong.

### 4.3 `run_srm_check`

- **Purpose:** Test whether control/treatment assignment matches the intended split ratio.
- **Parameters:** `{}` (no parameters)
- **State change:** Populates `observation.randomization_check` with `SRM_Result`.
- **When to use:** Should be run on **every** episode as a baseline check. The aggregate `control_count` and `treatment_count` fields provide the hint — if they don't match the intended 50/50 split, this action must be taken.
- **Failure modes:** Agent sees the user count asymmetry in aggregate results but jumps to a verdict without running the formal SRM check (evidence quality penalty).

### 4.4 `query_assignment_overlap`

- **Purpose:** Check if users in this experiment are simultaneously enrolled in other experiments in ways that could cause interference.
- **Parameters:** `{"target_experiment_id": str | None}` — None returns the full overlap matrix for all active experiments.
- **State change:** Populates `observation.user_assignment_overlap`.
- **When to use:** When `experiment_metadata` mentions other experiments running in the same time window, or when the targeting rule is narrow (increasing overlap probability).
- **Failure modes:** Agent fails to run this action on SUTVA violation episodes because the interference is not visible in any initial observation field.

### 4.5 `check_network_exposure`

- **Purpose:** Test whether treatment users' behavior leaked to control users via social graph connections.
- **Parameters:** `{}`
- **State change:** Populates `observation.network_exposure_map` — a dict mapping control user segments to their estimated exposure fraction.
- **When to use:** When the product being tested has social/network features (sharing, messaging, leaderboards).
- **Failure modes:** Agent ignores this on network spillover episodes because the experiment_metadata doesn't explicitly mention social features.

### 4.6 `query_secondary_metrics`

- **Purpose:** Look at guardrail metrics beyond the primary metric.
- **Parameters:** `{"metrics": List[str]}` or `{}` for all available secondary metrics.
- **State change:** Populates `observation.secondary_metric_results`.
- **When to use:** Whenever primary metric shows positive lift — guardrails may show harm to session length, DAU, or revenue per user.
- **Failure modes:** Agent approves a result without checking secondary metrics that show significant negative effects.

### 4.7 `compute_mde`

- **Purpose:** Verify that the experiment was adequately powered to detect the claimed effect size.
- **Parameters:** `{}`
- **State change:** Populates `observation.mde_analysis` with `MDEAnalysis`.
- **When to use:** When sample sizes seem small relative to the claimed effect, or when results are borderline significant.
- **Failure modes:** Agent doesn't compute MDE on the "underpowered_overclaim" episode because the p-value is below 0.05.

### 4.8 `inspect_randomization`

- **Purpose:** Audit the randomization algorithm, seed, and assignment log for implementation bugs.
- **Parameters:** `{}`
- **State change:** Populates `observation.randomization_audit`.
- **When to use:** When SRM is detected — follow up to understand the cause.
- **Failure modes:** Agent stops after detecting SRM and doesn't inspect randomization, missing the opportunity to cite the specific bug as evidence.

### 4.9 `flag_contamination` (Terminal)

- **Purpose:** Formally declare the experiment result invalid, specifying contamination type and evidence.
- **Parameters:** Full structured invalidation report (see Action model).
- **State change:** Sets `episode_done = True`.
- **When to use:** When investigation has revealed contamination evidence sufficient to explain the result.
- **Failure modes:** Agent flags with correct type but incorrect evidence facts (partial credit loss). Agent flags without running required investigative queries (investigation coverage penalty).

### 4.10 `approve_result` (Terminal)

- **Purpose:** Formally declare the result valid and shippable.
- **Parameters:** `{"reasoning": str, "confidence": float}`
- **State change:** Sets `episode_done = True`.
- **When to use:** ONLY on genuinely clean experiments after running key validity checks.
- **Failure modes:** Agent approves a contaminated experiment (maximum negative reward: -0.40).

### 4.11 `request_rerun` (Terminal)

- **Purpose:** Declare that the experiment design is flawed and needs to be restarted.
- **Parameters:** `{"reason": str, "recommended_changes": List[str]}`
- **State change:** Sets `episode_done = True`.
- **Scoring:** Partial credit — better than approving a contaminated result, worse than correct `flag_contamination` with accurate type identification.

---

## 5. Observation Design

### Initial Observation (Episode Start)

The agent receives:
- `experiment_id` — unique identifier (e.g., `"exp_2024_growth_042"`)
- `primary_metric` — what the experiment measured (e.g., `"D7 retention rate"`)
- `aggregate_results` — single-row summary table with counts, means, lift, p-value, CI
- `experiment_metadata` — dates, targeting, intended split, platform
- `available_queries` — full list of 8 investigative action types
- `steps_taken: 0`, `steps_remaining: 15`

**What is hidden at start:**
- The `ContaminationSpec` — never revealed
- All subgroup breakdowns
- Temporal breakdown
- SRM test result
- Assignment overlap
- Secondary metrics
- MDE analysis

### Information Revelation Principle

Each investigative action reveals exactly one new data category. The observation is **cumulative** — previously revealed data persists. The agent never loses information.

**Designed information hiding rationale:**
- Hiding subgroup data forces the agent to *decide* to break down results (Simpson's paradox detection skill).
- Hiding temporal data forces the agent to *decide* to look for novelty effects.
- Hiding SRM results forces the agent to *notice* the count asymmetry in aggregate results and take action.
- Hiding assignment overlap forces the agent to *hypothesize* that interference could be occurring.

This mirrors exactly how real analysts work — the data exists in the database, but you have to know to query it.

---

## 6. Environment Dynamics

### 6.1 `reset()` Flow

```python
def reset(task_id: int = 1, seed: int = 42) -> ExperimentObservation:
    # 1. Generate ContaminationSpec (hidden)
    spec = TaskGenerator.sample(task_id=task_id, seed=seed)
    
    # 2. Generate synthetic experiment data consistent with spec
    data = DataGenerator.generate(spec=spec, seed=seed)
    
    # 3. Initialize EpisodeState
    state = StateManager.init(
        experiment_id=data.experiment_id,
        spec=spec,
        data=data,
        max_steps=15
    )
    
    # 4. Build initial observation (only aggregate + metadata)
    obs = ObservationBuilder.build_initial(state, data)
    
    # 5. Log episode start
    Logger.log_episode_start(state.session_id, task_id, seed)
    
    return obs
```

### 6.2 `step()` Flow

```python
def step(action: AuditAction, session_id: str) -> StepResult:
    # 1. Validate action schema
    try:
        validated = AuditAction(**action.dict())
    except ValidationError as e:
        state.invalid_action_count += 1
        if state.invalid_action_count >= 3:
            return StepResult(done=True, termination="invalid_action_limit")
        return StepResult(error=str(e), done=False)
    
    # 2. Check duplicate query
    if validated.action_type in state.executed_queries:
        reward = RewardEngine.compute_duplicate_penalty()
        return StepResult(obs=current_obs, reward=reward, done=False)
    
    # 3. Execute action → reveal new data
    new_data = ActionExecutor.execute(validated, state)
    state.revealed_data[validated.action_type] = new_data
    state.executed_queries.append(validated.action_type)
    
    # 4. Check terminal condition
    done = validated.action_type in ["flag_contamination", "approve_result", "request_rerun"]
    
    # 5. Increment step count
    state.step_count += 1
    done = done or (state.step_count >= state.max_steps)
    
    # 6. Compute reward
    reward = RewardEngine.compute(validated, state, state.spec)
    
    # 7. Build updated observation
    obs = ObservationBuilder.build_updated(state, state.data)
    
    # 8. Log step
    Logger.log_step(session_id, validated, reward, obs)
    
    return StepResult(observation=obs, reward=reward.step_reward, done=done,
                      info={"cumulative_reward": reward.cumulative_reward})
```

### 6.3 `state()` Structure

```python
def state(session_id: str) -> Dict:
    return {
        "session_id": session_id,
        "step_count": state.step_count,
        "steps_remaining": state.max_steps - state.step_count,
        "executed_queries": state.executed_queries,
        "episode_done": state.episode_done,
        "cumulative_reward": state.cumulative_reward,
        # Note: ContaminationSpec is NEVER included here
    }
```

### 6.4 Episode Lifecycle

```
reset() → [Initial Observation]
  ↓
step(investigative_action_1) → [Observation + new data revealed]
  ↓
step(investigative_action_2) → [Observation + more data revealed]
  ↓
  ... (up to 15 steps total)
  ↓
step(terminal_action) → [Final Observation, done=True]
  ↓
Grader.grade(episode_log, spec) → Score [0.0, 1.0]
```

### 6.5 Termination Conditions

| Condition | `termination_reason` |
|---|---|
| Agent submits terminal action | `"agent_verdict"` |
| Step count reaches 15 | `"budget_exhausted"` (treated as `request_rerun` for scoring) |
| 3 consecutive invalid actions | `"invalid_action_limit"` (score = 0.0) |

---

## 7. Task Design (Easy → Hard)

### Task 1 — Easy: Sample Ratio Mismatch

**Scenario:**  
Experiment `exp_2024_growth_007` tests a new onboarding flow. Hypothesis: "Improved onboarding increases D7 retention." Primary metric: D7 retention rate.

**Initial observation shows:**
- `control_count: 61,847` | `treatment_count: 48,203`
- `relative_lift: +6.2%` | `p_value: 0.03`
- `intended_split: 0.50`

**Hidden ContaminationSpec:**
```python
ContaminationSpec(
    contamination_type="srm",
    true_effect_size=0.0,           # the measured effect is entirely an artifact
    visible_effect_size=0.062,
    srm_actual_split=0.438,         # treatment got only 43.8% of users
    optimal_investigation_steps=3,
    required_queries=["run_srm_check"],
    ground_truth_evidence={
        "control_count": 61847,
        "treatment_count": 48203,
        "actual_split": 0.438,
        "chi_square": 1847.3,
        "srm_p_value": 0.0000001
    }
)
```

**Expected agent steps:**
1. Observe the count asymmetry (61,847 vs 48,203 ≠ 50/50).
2. `run_srm_check` → SRM detected, chi-square = 1847.3, p < 0.001.
3. `flag_contamination(type="srm", evidence_facts=[count asymmetry, chi-square statistic])`.

**LLM failure trap:** Most LLMs see `p_value: 0.03` and `lift: +6.2%` and immediately issue `approve_result`. The count asymmetry is present in the aggregate results but requires arithmetic to notice (48203 / (48203 + 61847) = 43.8%, not 50%).

**Difficulty: Easy** — single contamination type, single required investigative action, all evidence visible in initial observation if read carefully.

---

### Task 2 — Medium: Simpson's Paradox + Secondary Metric Reversal

**Scenario:**  
Experiment `exp_2024_product_033` tests a redesigned home feed. Primary metric: D7 retention rate.

**Initial observation shows:**
- `control_count: 100,412` | `treatment_count: 99,588` ← balanced (no SRM hint)
- `relative_lift: +5.1%` | `p_value: 0.001` ← looks highly significant
- `intended_split: 0.50`
- `start_date: 2024-03-01` | `end_date: 2024-03-21`
- `targeting_rule: "all_mobile_users"`

**Hidden ContaminationSpec:**
```python
ContaminationSpec(
    contamination_type="simpsons_paradox",
    true_effect_size=-0.008,        # feature actually slightly hurts retention
    visible_effect_size=0.051,
    contaminated_subgroup="enrollment_cohort",
    optimal_investigation_steps=5,
    required_queries=["query_temporal", "query_subgroup", "query_secondary_metrics"],
    ground_truth_evidence={
        "early_cohort_baseline_retention": 0.71,  # days 1-3: high retention users
        "late_cohort_baseline_retention": 0.38,   # days 4-21: normal users
        "early_cohort_treatment_fraction": 0.67,  # treatment over-indexed early cohort
        "session_length_lift": -0.041,            # guardrail metric is negative
    }
)
```

**What each query reveals:**
- `query_temporal` → day 1–3 shows massive lift (+22%), day 4–21 shows near-zero (−1%). Reveals the temporal confound.
- `query_subgroup(dimension="enrollment_cohort")` → "days 1-3" cohort: control_mean=0.68, treatment_mean=0.71. "days 4-21" cohort: control_mean=0.39, treatment_mean=0.38. The aggregate is driven by compositional imbalance, not the treatment.
- `query_secondary_metrics` → session length shows −4.1% lift (statistically significant harm). Primary metric is positive, guardrail is negative.

**Expected agent steps:**
1. Run `query_temporal` → see the unusual day 1–3 spike.
2. Run `query_subgroup(dimension="enrollment_cohort")` → see the reversal.
3. Run `query_secondary_metrics` → confirm feature harms engagement.
4. (Optional) `inspect_randomization` → confirm the targeting rule change on day 4 explains the cohort imbalance.
5. `flag_contamination(type="simpsons_paradox", evidence_facts=[...])`.

**LLM failure trap 1:** Agent sees p=0.001 and approves immediately. **LLM failure trap 2:** Agent runs temporal query, sees strong early results, concludes novelty effect rather than Simpson's paradox. **LLM failure trap 3:** Agent flags Simpson's paradox but cites wrong subgroup (device instead of enrollment cohort).

**Difficulty: Medium** — requires 3 investigative actions, causal reasoning about cohort composition, and secondary metric awareness.

---

### Task 3 — Hard: SUTVA Violation + Network Spillover + Underpowered Overclaim

**Scenario:**  
Experiment `exp_2024_commerce_019` tests a new checkout flow. Primary metric: purchase conversion rate.

**Initial observation shows:**
- `control_count: 90,000` | `treatment_count: 90,000` ← balanced
- `relative_lift: +8.3%` | `p_value: 0.008` ← highly significant
- `intended_split: 0.50`
- `targeting_rule: "US users, purchased in last 90 days"`
- `start_date: 2024-02-01` | `end_date: 2024-02-28`

**Hidden ContaminationSpec (3 simultaneous layers):**
```python
ContaminationSpec(
    contamination_type="sutva_violation",  # primary contamination
    true_effect_size=0.012,               # actual effect: ~1.2%, not 8.3%
    visible_effect_size=0.083,
    interference_experiment_id="exp_2024_pricing_011",
    network_spillover_fraction=0.23,
    optimal_investigation_steps=8,
    required_queries=[
        "query_assignment_overlap",
        "check_network_exposure",
        "compute_mde"
    ],
    ground_truth_evidence={
        # Layer 1: SUTVA
        "control_users_in_pricing_treatment": 0.71,   # 71% of control also in pricing treatment
        "treatment_users_in_pricing_treatment": 0.28, # 28% of treatment in pricing treatment
        # Layer 2: Network spillover
        "control_users_exposed_to_treatment_behavior": 0.23,
        # Layer 3: Power
        "required_sample_for_8pct_effect": 400000,   # per arm
        "actual_sample_per_arm": 90000,
        "achieved_power": 0.21,                       # only 21% power
    }
)
```

**What each query reveals:**
- `query_assignment_overlap` → 71% of control users are also in the treatment arm of `exp_2024_pricing_011`. The pricing treatment increases purchase rates. Control is inflated → apparent lift of checkout feature is artificially high.
- `check_network_exposure` → 23% of control users have household connections to treatment users. They observed the new checkout flow indirectly.
- `compute_mde` → To detect an 8.3% effect at 80% power with baseline variance of purchase rate, you need 400K users per arm. Actual: 90K. Achieved power: 21%. The "significant" p-value resulted from early stopping after an unusually favorable 28-day window.

**Expected agent steps:**
1. `query_assignment_overlap` → discover the 71% SUTVA contamination.
2. `check_network_exposure` → discover the 23% spillover.
3. `compute_mde` → discover the experiment was 4.4× underpowered.
4. `query_secondary_metrics` → optional but adds evidence quality.
5. `flag_contamination` with: type="sutva_violation", all three evidence layers cited, `estimated_true_effect: 0.012`.

**Scoring breakdown for this task:**
- Correct terminal verdict: 0.35
- Correct primary type (sutva_violation): 0.25
- Investigation coverage (3/3 required queries): 0.20
- Evidence quality (all three layers cited): 0.12
- True effect estimate within 50% of actual: 0.08

**LLM failure trap 1:** Agent runs `query_assignment_overlap`, finds SUTVA, declares the result invalid, and stops — missing network spillover and power analysis (loses 0.20 on investigation + 0.12 on evidence). **LLM failure trap 2:** Agent flags `underpowered_overclaim` (one of three actual contaminations) — partial credit only.

**Difficulty: Hard** — requires 3+ investigative actions, multi-layer synthesis, and quantitative estimation.

---

### Task 4 — Expert (Optional): Clean Result with Red Herrings

**Scenario:**  
Experiment `exp_2024_search_055` tests a new search ranking algorithm. Primary metric: click-through rate.

**Initial observation shows:**
- `control_count: 250,000` | `treatment_count: 250,000`
- `relative_lift: +3.8%` | `p_value: 0.0002`
- `intended_split: 0.50`
- Temporal breakdown (if queried) shows a dip on day 4.
- Another experiment (`exp_2024_ads_014`) is listed in peer experiments.

**Hidden ContaminationSpec:**
```python
ContaminationSpec(
    contamination_type="clean",
    true_effect_size=0.038,
    visible_effect_size=0.038,
    optimal_investigation_steps=4,
    required_queries=["run_srm_check", "query_temporal", "query_assignment_overlap"]
)
```

**Red herrings engineered into the data:**
- Day 4 dip: caused by a documented platform outage (visible in `randomization_audit`), affects both arms equally, does not invalidate results.
- `exp_2024_ads_014` overlap: appears to affect the same users but uses a different `randomization_unit` (session-level vs user-level), so there is no actual SUTVA contamination.

**Expected agent steps:**
1. `run_srm_check` → no SRM detected.
2. `query_temporal` → sees day 4 dip, investigates.
3. `inspect_randomization` → sees outage log on day 4, confirms it affected both arms symmetrically.
4. `query_assignment_overlap` → sees ads experiment but notices different randomization units.
5. `approve_result(confidence=0.85)`.

**LLM failure trap:** Agent sees the day 4 dip and flags `novelty_effect` (wrong — that's a platform event). Agent sees the overlapping experiment and flags `sutva_violation` (wrong — different randomization units prevent interference). Score = 0.0 for over-triggering on red herrings.

**Difficulty: Expert** — tests false-positive avoidance, requires understanding of platform outage context, and demands nuanced reasoning about randomization unit compatibility.

---

## 8. Reward Function

### Complete Reward Formula

```python
def compute_reward(action: AuditAction, state: EpisodeState, spec: ContaminationSpec) -> float:
    r = 0.0

    # ─────────────────────────────────────────────
    # INVESTIGATIVE REWARDS (step-level, dense)
    # ─────────────────────────────────────────────

    INVESTIGATIVE_ACTIONS = [
        "query_subgroup", "query_temporal", "run_srm_check",
        "query_assignment_overlap", "check_network_exposure",
        "inspect_randomization", "query_secondary_metrics", "compute_mde"
    ]

    if action.action_type in INVESTIGATIVE_ACTIONS:
        # Reward relevant investigation
        if _query_is_relevant(action, spec):
            r += 0.08   # investigating something informative for this contamination type
            if _query_reveals_signal(action, spec, state):
                r += 0.06   # and it actually found a contamination signal
        else:
            r -= 0.03   # wasted action budget on irrelevant query

        # Penalty for redundant query
        if action.action_type in state.executed_queries:
            r -= 0.03   # already ran this query

    # ─────────────────────────────────────────────
    # CALIBRATION REWARD (available on any action with confidence)
    # ─────────────────────────────────────────────
    if action.confidence is not None:
        evidence_strength = _compute_evidence_strength(state, spec)
        calibration_error = abs(action.confidence - evidence_strength)
        calibration_score = max(0, 1.0 - calibration_error * 2)
        r += 0.05 * calibration_score   # small step-level calibration signal

    # ─────────────────────────────────────────────
    # TERMINAL REWARDS
    # ─────────────────────────────────────────────

    if action.action_type == "flag_contamination":
        claimed_type = action.parameters.get("contamination_type")

        # Primary: verdict correctness
        if spec.contamination_type == "clean":
            r -= 0.40   # false positive — worst outcome
        else:
            # Contamination type match
            type_score = TYPE_MATCH_MATRIX.get(
                (claimed_type, spec.contamination_type), 0.0
            )
            r += 0.30 * type_score   # max +0.30 for exact type match

            # Evidence quality
            evidence_facts = action.parameters.get("evidence_facts", [])
            evidence_score = _score_evidence_facts(evidence_facts, spec)
            r += 0.20 * evidence_score   # max +0.20 for accurate factual evidence

            # Quantitative estimate (Tasks 3+)
            if "estimated_true_effect" in action.parameters and spec.true_effect_size != spec.visible_effect_size:
                estimate = action.parameters["estimated_true_effect"]
                range_size = abs(spec.visible_effect_size - spec.true_effect_size) + 1e-6
                estimate_accuracy = max(0, 1 - abs(estimate - spec.true_effect_size) / range_size)
                r += 0.15 * estimate_accuracy   # max +0.15 for accurate effect estimate

            # Terminal calibration
            terminal_calibration = 1.0 - abs(action.confidence - _compute_evidence_strength(state, spec))
            r += 0.10 * terminal_calibration   # max +0.10

    if action.action_type == "approve_result":
        if spec.contamination_type == "clean":
            r += 0.35   # correctly approved a valid experiment
            terminal_calibration = 1.0 - abs(action.confidence - 0.9)
            r += 0.10 * terminal_calibration
        else:
            r -= 0.40   # approved a contaminated experiment — maximum negative reward

    if action.action_type == "request_rerun":
        if spec.contamination_type != "clean":
            r += 0.10   # partial credit — at least didn't approve a bad result
        else:
            r -= 0.10   # unnecessary rerun of a valid experiment

    # ─────────────────────────────────────────────
    # EFFICIENCY PENALTY (end of episode)
    # ─────────────────────────────────────────────
    if state.episode_done:
        steps_over_optimal = max(0, state.step_count - spec.optimal_investigation_steps * 2)
        r -= 0.03 * steps_over_optimal   # cap at ~-0.30 for very inefficient agents

    return round(r, 4)
```

### Why Each Reward Component Exists

| Component | Value | Behavioral Intent |
|---|---|---|
| Relevant investigative query | +0.08 | Reward hypothesis-driven investigation |
| Query reveals contamination signal | +0.06 | Reward finding what you were looking for |
| Irrelevant query | −0.03 | Discourage blind enumeration of all possible queries |
| Duplicate query | −0.03 | Discourage repetitive behavior |
| Step-level calibration | +0.05 | Reward confidence tracking throughout episode |
| Correct terminal type (exact) | +0.30 | Primary success metric |
| Evidence quality | +0.20 | Reward structured factual reasoning |
| Quantitative effect estimate | +0.15 | Reward numerical reasoning |
| Terminal calibration | +0.10 | Reward appropriate confidence at verdict |
| False positive (flagging clean) | −0.40 | Maximum penalty — as bad as approving contaminated |
| Approving contaminated | −0.40 | Maximum penalty — the worst real-world outcome |
| Request rerun on contaminated | +0.10 | Partial credit for caution over false approval |
| Efficiency over-budget | −0.03/step | Discourage excessive investigation |

### Reward Range Analysis

- **Perfect episode (Task 1):** ~0.85 (0.08 + 0.06 for SRM query + 0.30 + 0.20 + 0.10 + 0.10 calibration, no efficiency penalty)
- **Typical successful episode:** 0.55–0.75
- **Approved contaminated experiment:** < 0.0 (heavy terminal penalty dominates)
- **Budget exhausted without verdict:** 0.15–0.35 (investigative rewards only, efficiency penalty)

---

## 9. Grader Design

```python
def grade_episode(episode_log: List[Dict], spec: ContaminationSpec) -> Dict:
    """
    Deterministic episode grader. No LLM involvement.
    Returns score in [0.0, 1.0].
    """
    scores = {}

    # ─── 1. VERDICT CORRECTNESS (weight: 0.35) ───
    # Binary: did the agent make the right final call?
    final_action = episode_log[-1]
    verdict = final_action["action_type"]

    if spec.contamination_type == "clean":
        scores["verdict"] = 1.0 if verdict == "approve_result" else 0.0
    else:
        scores["verdict"] = 1.0 if verdict == "flag_contamination" else 0.0

    # ─── 2. CONTAMINATION TYPE IDENTIFICATION (weight: 0.25) ───
    # Partial credit matrix for related types
    TYPE_MATCH_MATRIX = {
        ("srm", "srm"): 1.0,
        ("srm", "underpowered_overclaim"): 0.3,           # adjacent concept
        ("underpowered_overclaim", "srm"): 0.3,
        ("sutva_violation", "network_spillover"): 0.5,    # mechanistically related
        ("network_spillover", "sutva_violation"): 0.5,
        ("simpsons_paradox", "multiple_testing"): 0.2,   # surface similarity
        ("novelty_effect", "simpsons_paradox"): 0.1,     # temporal confound family
        # All unlisted pairs → 0.0
    }

    if verdict == "flag_contamination":
        claimed_type = final_action["parameters"].get("contamination_type", "")
        scores["type_id"] = TYPE_MATCH_MATRIX.get(
            (claimed_type, spec.contamination_type),
            1.0 if claimed_type == spec.contamination_type else 0.0
        )
    else:
        scores["type_id"] = 0.0

    # ─── 3. INVESTIGATION COVERAGE (weight: 0.20) ───
    # Did the agent run the queries needed to find the contamination?
    required = set(spec.required_queries or [])
    executed = set(a["action_type"] for a in episode_log)
    if required:
        scores["investigation"] = len(required & executed) / len(required)
    else:
        scores["investigation"] = 1.0   # clean experiment: any validation is credit

    # ─── 4. EVIDENCE QUALITY (weight: 0.12) ───
    # Are the facts cited in the flag_contamination action actually correct?
    if verdict == "flag_contamination":
        evidence_facts = final_action["parameters"].get("evidence_facts", [])
        scores["evidence"] = _verify_evidence_facts(evidence_facts, spec)
    else:
        scores["evidence"] = 0.0

    # _verify_evidence_facts checks each claimed fact against ground_truth_evidence
    # in the ContaminationSpec. Returns fraction of facts that are numerically
    # accurate (within 10% tolerance for continuous values, exact for categorical).

    # ─── 5. CALIBRATION (weight: 0.08) ───
    # Correlation between agent's stated confidence and actual evidence strength
    confidence_actions = [a for a in episode_log if "confidence" in a and a.get("confidence") is not None]

    if len(confidence_actions) >= 2:
        confidences = [a["confidence"] for a in confidence_actions]
        # evidence_strength is computed from queries executed at each step
        evidence_strengths = []
        for i, action in enumerate(confidence_actions):
            step_idx = episode_log.index(action)
            queries_so_far = [a["action_type"] for a in episode_log[:step_idx+1]]
            evidence_strengths.append(_evidence_strength_at_step(queries_so_far, spec))

        calibration_errors = [abs(c - e) for c, e in zip(confidences, evidence_strengths)]
        scores["calibration"] = max(0.0, 1.0 - (sum(calibration_errors) / len(calibration_errors)) * 2)
    else:
        scores["calibration"] = 0.5   # neutral if confidence rarely stated

    # ─── FINAL WEIGHTED SCORE ───
    weights = {
        "verdict": 0.35,
        "type_id": 0.25,
        "investigation": 0.20,
        "evidence": 0.12,
        "calibration": 0.08
    }

    final_score = sum(scores[k] * weights[k] for k in weights)
    return {
        "final_score": round(final_score, 4),
        "breakdown": scores,
        "weights": weights,
        "verdict_action": verdict,
        "ground_truth_type": spec.contamination_type
    }
```

### Evidence Verification Logic

```python
def _verify_evidence_facts(claimed_facts: List[str], spec: ContaminationSpec) -> float:
    """
    Checks each fact in evidence_facts against the ground truth.
    Returns fraction of verifiable facts that are correct.
    """
    if not claimed_facts:
        return 0.0

    verified = 0
    checkable = 0
    gt = spec.ground_truth_evidence or {}

    # For each claimed fact, attempt to extract a numeric or categorical claim
    # and match it against the ground truth dictionary
    for fact in claimed_facts:
        for key, gt_value in gt.items():
            if key.replace("_", " ") in fact.lower():
                checkable += 1
                # Extract numeric from fact string (simple regex)
                nums = re.findall(r"[-+]?\d*\.?\d+", fact)
                if nums and isinstance(gt_value, float):
                    claimed_val = float(nums[-1]) / (100 if "%" in fact else 1)
                    if abs(claimed_val - gt_value) / (abs(gt_value) + 1e-6) <= 0.10:
                        verified += 1
                elif isinstance(gt_value, str) and gt_value.lower() in fact.lower():
                    verified += 1

    return verified / max(checkable, 1)
```

---

## 10. Baseline Agent Design

### Strategy: LLM-Driven with Structured Prompting

The baseline agent uses a GPT-class model (via OpenAI client with `API_BASE_URL` override) with a structured prompt that:
1. Explains the audit task and action space.
2. Shows the current observation.
3. Requests a single action in JSON format.

### System Prompt

```
You are an expert A/B test validity auditor at a large technology company.

Your task: Given an experiment summary, investigate whether the results are statistically valid or contaminated by hidden biases. 

You MUST:
1. Check for Sample Ratio Mismatch (SRM) before trusting any results.
2. Break down results by subgroup if the aggregate looks unusually clean or strong.
3. Check temporal trends for novelty effects.
4. Examine secondary/guardrail metrics — positive primary metric with negative guardrail = suspicious.
5. Check for experiment overlap when multiple experiments are running simultaneously.
6. Verify the experiment was adequately powered for the claimed effect size.

You must NOT:
- Approve an experiment without running at least one validity check.
- Ignore the user count in aggregate_results.
- Flag a clean experiment just because something looks slightly unusual.

Available actions: query_subgroup, query_temporal, run_srm_check, query_assignment_overlap,
check_network_exposure, inspect_randomization, query_secondary_metrics, compute_mde,
flag_contamination, approve_result, request_rerun

Respond ONLY with a valid JSON object matching this schema:
{
  "action_type": "<action name>",
  "parameters": {<action-specific params>},
  "reasoning": "<your reasoning, 1-3 sentences>",
  "confidence": <float 0.0-1.0>
}
```

### Action Selection Loop

```python
def run_episode(env, task_id: int, seed: int, client: OpenAI) -> Dict:
    obs = env.reset(task_id=task_id, seed=seed)
    episode_log = []
    total_reward = 0.0

    for step in range(15):
        prompt = format_observation_as_prompt(obs)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,    # low temperature for deterministic baseline
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        action_json = json.loads(response.choices[0].message.content)
        action = AuditAction(**action_json)
        
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        episode_log.append({**action_json, "reward": result.reward})
        
        if result.done:
            break
    
    return {
        "episode_log": episode_log,
        "total_reward": total_reward,
        "steps_taken": len(episode_log)
    }
```

### Handling Uncertainty

- If JSON parsing fails: retry once with explicit format correction in the prompt.
- If action validation fails: log the error, substitute `request_rerun` as fallback.
- If API call fails: exponential backoff, max 3 retries.

### Expected Baseline Scores

| Task | Expected Score | Notes |
|---|---|---|
| Task 1 (Easy) | 0.55–0.70 | Most LLMs miss the SRM unless explicitly prompted to check counts |
| Task 2 (Medium) | 0.35–0.55 | Temporal + subgroup synthesis is hard for one-step reasoners |
| Task 3 (Hard) | 0.15–0.35 | Multi-layer contamination almost always partially missed |
| Task 4 (Expert) | 0.40–0.60 | Clean result with red herrings — depends on false-positive behavior |

---

## 11. Inference Script (`inference.py`)

```python
"""
Inference Script — A/B Test Causal Contamination Forensics
===========================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key
"""

import os
import json
import time
import logging
from typing import Dict, List

from openai import OpenAI

# ── Local imports (environment runs as a package) ──
from env.core import ABTestContaminationEnv
from env.models import AuditAction

# ── Configuration from environment variables ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-70B-Instruct")

MAX_STEPS    = 12
TEMPERATURE  = 0.1
MAX_TOKENS   = 500
TASKS        = [1, 2, 3]          # Tasks to evaluate
SEEDS        = [42, 123, 7]       # One seed per task for reproducibility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert A/B test validity auditor. Given an experiment briefing, 
investigate whether results are statistically valid or contaminated.

Rules:
1. Always run run_srm_check early — count imbalance invalidates everything.
2. Check temporal trends if the experiment ran >7 days.
3. Check subgroups when aggregate lift seems unusually high.
4. Check secondary metrics before approving anything.
5. Check experiment overlap when peer experiments exist.
6. Use compute_mde when sample sizes seem small for the claimed effect.

Respond ONLY with a valid JSON object:
{
  "action_type": "<one of the available actions>",
  "parameters": {},
  "reasoning": "<1-3 sentence justification>",
  "confidence": <float 0.0-1.0>
}
""".strip()


def format_obs(obs: Dict) -> str:
    """Format observation dict into a clear text prompt for the LLM."""
    lines = [
        f"=== EXPERIMENT AUDIT: {obs.get('experiment_id', 'UNKNOWN')} ===",
        f"Primary metric: {obs.get('primary_metric', '')}",
        f"Hypothesis: {obs.get('experiment_metadata', {}).get('hypothesis', '')}",
        "",
        "AGGREGATE RESULTS:",
        f"  Control:   n={obs['aggregate_results']['control_count']:,}  "
        f"mean={obs['aggregate_results']['control_mean']:.4f}",
        f"  Treatment: n={obs['aggregate_results']['treatment_count']:,}  "
        f"mean={obs['aggregate_results']['treatment_mean']:.4f}",
        f"  Lift: {obs['aggregate_results']['relative_lift']*100:.2f}%  "
        f"p={obs['aggregate_results']['p_value']:.4f}",
        "",
        f"METADATA:",
        f"  Date: {obs['experiment_metadata']['start_date']} → {obs['experiment_metadata']['end_date']}",
        f"  Intended split: {obs['experiment_metadata']['intended_split']*100:.0f}% treatment",
        f"  Targeting: {obs['experiment_metadata']['targeting_rule']}",
        "",
        f"Steps remaining: {obs.get('steps_remaining', '?')}",
        f"Available actions: {', '.join(obs.get('available_queries', []))}",
    ]

    # Append any revealed data
    for field in ["randomization_check", "subgroup_results", "temporal_breakdown",
                  "user_assignment_overlap", "secondary_metric_results", "mde_analysis",
                  "network_exposure_map"]:
        if obs.get(field):
            lines.append(f"\n[REVEALED] {field.upper()}:")
            lines.append(json.dumps(obs[field], indent=2, default=str)[:1500])  # truncate large payloads

    return "\n".join(lines)


def run_episode(env: ABTestContaminationEnv, task_id: int, seed: int,
                client: OpenAI) -> Dict:
    """Run one complete episode and return results."""
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.dict()
    episode_log = []
    total_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    log.info(f"Starting Task {task_id} | Seed {seed} | Exp {obs_dict['experiment_id']}")

    for step_num in range(MAX_STEPS):
        user_msg = format_obs(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        # ── LLM call ──
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"}
                )
                action_str = response.choices[0].message.content
                action_json = json.loads(action_str)
                break
            except Exception as e:
                log.warning(f"LLM call attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
                if attempt == 2:
                    action_json = {"action_type": "request_rerun",
                                   "parameters": {}, "reasoning": "API error fallback",
                                   "confidence": 0.1}

        # Validate and execute action
        try:
            action = AuditAction(**action_json)
        except Exception as e:
            log.warning(f"Invalid action format: {e}. Using fallback.")
            action = AuditAction(action_type="request_rerun", parameters={},
                                 reasoning="Malformed action fallback", confidence=0.1)

        result = env.step(action=action)
        obs_dict = result.observation.dict()
        total_reward += result.reward
        episode_log.append({
            "step": step_num + 1,
            "action": action.dict(),
            "reward": result.reward,
            "done": result.done
        })

        # Add assistant turn to messages for multi-turn context
        messages.append({"role": "assistant", "content": action_str})

        log.info(f"  Step {step_num+1}: {action.action_type} | reward={result.reward:.3f}")

        if result.done:
            log.info(f"  Episode done at step {step_num+1}")
            break

    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": round(total_reward, 4),
        "steps_taken": len(episode_log),
        "episode_log": episode_log,
        "final_action": episode_log[-1]["action"]["action_type"] if episode_log else None
    }


def main():
    log.info(f"Model: {MODEL_NAME} | API: {API_BASE_URL}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ABTestContaminationEnv()

    all_results = []
    scores_by_task = {}

    for task_id, seed in zip(TASKS, SEEDS):
        result = run_episode(env, task_id=task_id, seed=seed, client=client)
        grade = env.grade_last_episode()          # calls Grader.grade() on the session

        result["grade"] = grade
        all_results.append(result)
        scores_by_task[f"task_{task_id}"] = grade["final_score"]

        log.info(f"Task {task_id} Score: {grade['final_score']:.4f} | "
                 f"Breakdown: {grade['breakdown']}")

    # ── Summary ──
    avg_score = sum(scores_by_task.values()) / len(scores_by_task)
    summary = {
        "model": MODEL_NAME,
        "average_score": round(avg_score, 4),
        "scores_by_task": scores_by_task,
        "episodes": all_results
    }

    with open("baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 50)
    log.info(f"BASELINE RESULTS")
    log.info(f"  Task 1 (Easy):   {scores_by_task.get('task_1', 0):.4f}")
    log.info(f"  Task 2 (Medium): {scores_by_task.get('task_2', 0):.4f}")
    log.info(f"  Task 3 (Hard):   {scores_by_task.get('task_3', 0):.4f}")
    log.info(f"  Average:         {avg_score:.4f}")
    log.info("Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
```

---

## 12. OpenEnv Compliance

### `openenv.yaml`

```yaml
name: ab-test-contamination-forensics
version: "1.0.0"
description: >
  An experimentation auditing environment where agents must identify
  statistical contamination in A/B test results. Agents audit validity,
  not metrics — detecting SRM, Simpson's paradox, SUTVA violations,
  novelty effects, and more.

author: "Your Name"
license: MIT
tags:
  - openenv
  - experimentation
  - causal-reasoning
  - statistics
  - forensics
  - real-world

observation_space:
  type: structured
  schema: ExperimentObservation
  progressive_disclosure: true

action_space:
  type: discrete_structured
  schema: AuditAction
  n_actions: 11
  terminal_actions:
    - flag_contamination
    - approve_result
    - request_rerun

reward:
  range: [-0.50, 1.00]
  dense: true
  sparse_terminal: true
  calibration_signal: true

tasks:
  - id: 1
    name: sample_ratio_mismatch
    difficulty: easy
    description: Detect SRM in a single contaminated experiment.
  - id: 2
    name: simpsons_paradox_novelty
    difficulty: medium
    description: Detect Simpson's paradox + secondary metric reversal.
  - id: 3
    name: multi_layer_contamination
    difficulty: hard
    description: Detect 3 simultaneous contamination mechanisms.
  - id: 4
    name: clean_with_red_herrings
    difficulty: expert
    description: Correctly approve a valid experiment despite misleading signals.

episode:
  max_steps: 15
  termination: terminal_action | budget_exhausted | invalid_action_limit

api:
  reset: POST /reset
  step: POST /step
  state: GET /state
  tasks: GET /tasks
  health: GET /health

inference:
  script: inference.py
  env_vars:
    - API_BASE_URL
    - MODEL_NAME
    - HF_TOKEN
  runtime_limit_minutes: 20
  reproducible: true
  seeds: [42, 123, 7]
```

### Validation Checklist

- [ ] `openenv validate` passes on `openenv.yaml`
- [ ] `POST /reset` returns valid `ExperimentObservation` JSON
- [ ] `POST /step` returns valid `StepResult` JSON with reward in [-1, 1]
- [ ] `GET /state` returns session state without exposing `ContaminationSpec`
- [ ] All 3 tasks run end-to-end without error
- [ ] Grader returns score in [0.0, 1.0] for all episodes
- [ ] Same seed always produces same data (determinism check)
- [ ] `inference.py` completes in < 20 minutes
- [ ] `docker build && docker run` succeeds

---

## 13. Project Structure

```
ab-test-contamination-env/
│
├── env/
│   ├── __init__.py
│   ├── core.py                 # ABTestContaminationEnv main class
│   ├── state_manager.py        # EpisodeState management
│   ├── observation_builder.py  # Observation composition logic
│   ├── action_executor.py      # Action validation and execution
│   ├── reward_engine.py        # Reward computation
│   └── data_generator.py       # Synthetic data generation from ContaminationSpec
│
├── models/
│   ├── __init__.py
│   ├── observation.py          # ExperimentObservation, AggregateResult, etc.
│   ├── action.py               # AuditAction
│   ├── reward.py               # StepReward, EpisodeReward
│   └── contamination_spec.py   # ContaminationSpec (hidden state)
│
├── tasks/
│   ├── __init__.py
│   ├── task_generator.py       # TaskGenerator.sample()
│   ├── task_1_srm.py           # Task 1 spec variants
│   ├── task_2_simpsons.py      # Task 2 spec variants
│   ├── task_3_multilayer.py    # Task 3 spec variants
│   └── task_4_clean.py         # Task 4 spec variants
│
├── grader/
│   ├── __init__.py
│   ├── grader.py               # grade_episode() main function
│   ├── evidence_verifier.py    # _verify_evidence_facts()
│   └── type_match_matrix.py    # Partial credit matrix for contamination types
│
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── routes.py               # /reset, /step, /state, /tasks, /health
│   └── session_store.py        # In-memory session management
│
├── logs/
│   └── .gitkeep
│
├── tests/
│   ├── test_env.py
│   ├── test_grader.py
│   ├── test_tasks.py
│   └── test_api.py
│
├── inference.py                # Baseline inference script (MANDATORY)
├── openenv.yaml                # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 14. Docker + Deployment

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose FastAPI port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

### `requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
openai==1.30.1
numpy==1.26.4
scipy==1.13.0
httpx==0.27.0
pytest==8.2.0
pytest-asyncio==0.23.7
```

### HF Space Deployment Steps

1. Create a new Space on Hugging Face: `New Space → Docker → SDK: Docker`
2. Set Space hardware: CPU Basic (2 vCPU, 16 GB RAM)
3. Add Space secrets (Settings → Repository secrets):
   ```
   API_BASE_URL = https://router.huggingface.co/v1
   MODEL_NAME   = meta-llama/Llama-3.1-70B-Instruct
   HF_TOKEN     = hf_...
   ```
4. Push code:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ab-test-contamination-env
   git push space main
   ```
5. Verify deployment: Space URL should return `{"status": "ok"}` at `/health`.
6. Tag the space with `openenv` in the Space metadata.

### Runtime Constraints (2 vCPU, 8 GB RAM)

- All data generation is CPU-only (numpy/scipy), no ML model inference in the environment.
- Session store is in-memory (dict), suitable for single-process deployment.
- Episode data is small (<10 KB per session).
- The inference script runs sequentially (3 episodes) and completes in <5 minutes for most models.
- If RAM is tight: reduce `SEEDS` to 1 per task and limit `MAX_STEPS` to 10.

---

## 15. Development Phase Plan

### Phase 1 — Foundation (Days 1–2)

**Goals:** Working skeleton with valid OpenEnv spec and API.

**Tasks:**
1. Initialize git repo and HF Space.
2. Create `requirements.txt` with all dependencies.
3. Define all Pydantic models in `models/` — start with `AuditAction` and `ExperimentObservation`.
4. Implement `ContaminationSpec` dataclass with all 8 contamination types.
5. Write stub implementations of `reset()`, `step()`, `state()` that return hardcoded data.
6. Create `openenv.yaml`.
7. Stand up FastAPI app with all 5 endpoints.
8. Write `Dockerfile` and verify `docker build`.

**Expected outputs:** `GET /health` returns 200. `POST /reset` returns a valid (hardcoded) observation.

**Common mistakes:**
- Not setting `response_format={"type": "json_object"}` in OpenAI calls.
- Pydantic v1 vs v2 API differences (use `model.dict()` → `model.model_dump()` in v2).
- FastAPI session isolation — use `session_id` as key, not global state.

**Validation checklist:**
- [ ] `docker build` succeeds.
- [ ] `GET /health` returns `{"status": "ok"}`.
- [ ] `openenv validate openenv.yaml` passes.

---

### Phase 2 — Core Environment (Days 3–5)

**Goals:** Real data generation, state management, progressive disclosure.

**Tasks:**
1. Implement `DataGenerator.generate(spec, seed)` — produces internally consistent synthetic experiment data.
2. Implement `StateManager` with `EpisodeState` and all mutable fields.
3. Implement `ObservationBuilder` with progressive unlock logic.
4. Implement `ActionExecutor` for all 8 investigative actions.
5. Wire `reset()` to call `TaskGenerator.sample()` → `DataGenerator` → `StateManager.init()` → `ObservationBuilder.build_initial()`.
6. Wire `step()` with the full flow documented in Section 6.2.

**Common mistakes:**
- Generating data without the seed — always pass `random.seed(seed)` before numpy operations.
- Not copying the spec to state — must be stored per-session, not as module-level global.
- Revealing data in the initial observation — always start with the minimal set.

**Validation checklist:**
- [ ] Same seed always produces same data (run reset() twice with same seed, compare).
- [ ] `step(run_srm_check)` reveals `randomization_check` in next observation.
- [ ] `step(query_temporal)` reveals `temporal_breakdown`.
- [ ] Duplicate query returns cached data without new information.

---

### Phase 3 — Task System (Days 5–7)

**Goals:** All 4 tasks with deterministic specs and data.

**Tasks:**
1. Implement Task 1 (SRM) — 3 spec variants, verify count asymmetry is visible in aggregate_results.
2. Implement Task 2 (Simpson's) — generate subgroup data showing the reversal, temporal data showing the cohort effect.
3. Implement Task 3 (Multi-layer) — generate overlap matrix, network exposure map, MDE analysis.
4. Implement Task 4 (Clean + red herrings) — generate day 4 dip with outage metadata, non-interfering peer experiment.
5. Write `TaskGenerator.sample(task_id, seed)` dispatch.

**Common mistakes:**
- Task 3 overlap matrix: the asymmetry (71% of control in Y-treatment, 28% of treatment) must be computed correctly from the cross-assignment fractions.
- Task 4 red herring: the peer experiment must have `randomization_unit="session_id"` to make it legitimately non-interfering.

**Validation checklist:**
- [ ] Task 1 `aggregate_results` always shows ~44/56 split (not 50/50).
- [ ] Task 2 `query_temporal` always shows day 1-3 spike.
- [ ] Task 3 `query_assignment_overlap` always shows 71% overlap.
- [ ] Task 4 `randomization_check` returns `srm_detected: False`.

---

### Phase 4 — Reward + Grader (Days 7–9)

**Goals:** Fully functional reward engine and deterministic grader.

**Tasks:**
1. Implement `RewardEngine.compute()` with all components.
2. Implement `_query_is_relevant()` for each contamination type.
3. Implement `_compute_evidence_strength()` based on queries executed.
4. Implement `Grader.grade_episode()` with all 5 scoring dimensions.
5. Implement `_verify_evidence_facts()` with numeric tolerance matching.
6. Implement `TYPE_MATCH_MATRIX` with all partial credit pairs.
7. Write unit tests for grader on handcrafted episode logs.

**Common mistakes:**
- Evidence verification is too strict (rejects facts expressed with slight wording variations). Use keyword-in-substring matching, not exact string equality.
- Calibration score computed incorrectly when agent never states confidence — default to 0.5, not 0.0.
- Reward accumulation: ensure cumulative reward is tracked in `EpisodeState`, not recomputed from scratch.

**Validation checklist:**
- [ ] Perfect Task 1 episode scores > 0.80.
- [ ] Approving a contaminated experiment produces total score < 0.20.
- [ ] Grader score is in [0.0, 1.0] for all episode types.
- [ ] Same episode log always produces same grade (determinism).

---

### Phase 5 — Baseline Agent (Days 9–10)

**Goals:** Working `inference.py` that completes in <20 minutes.

**Tasks:**
1. Write `format_obs()` function — converts observation dict to readable text prompt.
2. Write `run_episode()` with full step loop and error handling.
3. Write `main()` with all 3 tasks, result aggregation, and JSON output.
4. Test with a real model call (use gpt-4o-mini or a small HF model).
5. Verify `baseline_results.json` is written correctly.

**Common mistakes:**
- Not using `response_format={"type": "json_object"}` — will cause JSON parse errors.
- Truncating the observation too aggressively — agent needs to see all revealed data.
- Not including the assistant's previous action in the message history — causes the agent to repeat actions.

**Validation checklist:**
- [ ] `python inference.py` completes without error.
- [ ] `baseline_results.json` contains scores for all 3 tasks.
- [ ] Same run produces same scores (low temperature + fixed seed).

---

### Phase 6 — Deployment (Days 10–11)

**Tasks:**
1. Push to HF Space, verify build completes.
2. Verify `/health` endpoint returns 200.
3. Run `openenv validate` against the live HF Space URL.
4. Run `inference.py` against the live environment (not localhost).
5. Write `README.md` with all required sections.

**Validation checklist:**
- [ ] HF Space URL returns `{"status": "ok"}`.
- [ ] `openenv validate https://YOUR-SPACE.hf.space` passes.
- [ ] Inference script runs against live URL in <20 minutes.

---

### Phase 7 — Testing & Validation (Day 12)

**Tasks:**
1. Run all pytest tests.
2. Verify determinism: run each task 3× with same seed, compare scores.
3. Verify grader range: generate 100 random episodes, assert all scores in [0.0, 1.0].
4. Run the full pre-submission checklist.

---

## 16. Failure Modes & Debugging Guide

### Agent-Side Failures

| Failure | Symptom | Fix |
|---|---|---|
| LLM approves contaminated experiment | Score = 0.0, heavy negative reward | Strengthen system prompt: "ALWAYS run at least one validity check before approving" |
| LLM flags clean experiment | Score = 0.0 on Task 4 | Add false-positive warning to system prompt |
| LLM repeats same query | Duplicate penalty accumulates | Add "do not repeat queries already executed" to prompt |
| JSON parse error from LLM | `json.JSONDecodeError` | Use `response_format={"type": "json_object"}` |
| LLM missing `reasoning` field | `ValidationError` | Add field description to prompt |

### Environment-Side Failures

| Failure | Symptom | Fix |
|---|---|---|
| Non-deterministic data | Different data on same seed | Ensure `random.seed(seed)` called before every numpy operation |
| Session bleed | Agent A sees Agent B's state | Use `session_id` dict key isolation, never global state |
| Grader returns > 1.0 | Score out of range | Add `min(1.0, max(0.0, final_score))` clamp |
| Reward accumulation drift | Cumulative reward doesn't match sum of step rewards | Track `cumulative_reward` in `EpisodeState`, update atomically |
| TaskGenerator incorrect for task_id | Wrong contamination type for task | Add assertion: `assert spec.contamination_type == EXPECTED_TYPES[task_id]` |
| Action executor not updating state | Query run but data not revealed | Verify `state.revealed_data[action_type] = new_data` is called |

### Reward Signal Issues

| Issue | Cause | Fix |
|---|---|---|
| Agent learns to always `request_rerun` | Positive partial credit higher than risk of negative terminal | Reduce `request_rerun` credit to +0.05 or add budget penalty |
| Agent loops on investigative actions | Dense rewards dominate terminal | Add stronger terminal rewards or hard step limit |
| Calibration reward noise | `_compute_evidence_strength` not monotone | Ensure evidence strength only increases as more queries are run |

---

## 17. Evaluation Strategy

### Correctness Testing

```bash
# Run full test suite
pytest tests/ -v

# Test specific components
pytest tests/test_grader.py -v
pytest tests/test_tasks.py -v -k "task_1"
```

### Determinism Validation

```python
# Run in tests/test_determinism.py
def test_determinism():
    env = ABTestContaminationEnv()
    for task_id in [1, 2, 3]:
        for seed in [42, 100, 999]:
            obs1 = env.reset(task_id=task_id, seed=seed)
            obs2 = env.reset(task_id=task_id, seed=seed)
            assert obs1.dict() == obs2.dict(), f"Non-deterministic: task={task_id} seed={seed}"
```

### Grader Range Validation

```python
# Generate 200 random episodes and assert all scores in [0.0, 1.0]
def test_grader_range():
    for _ in range(200):
        episode_log = generate_random_episode_log()
        spec = random.choice(ALL_SPECS)
        result = grade_episode(episode_log, spec)
        assert 0.0 <= result["final_score"] <= 1.0
```

### Task Difficulty Ordering

```python
# Verify that random agent scores Task 1 > Task 2 > Task 3
def test_difficulty_ordering():
    random_agent_scores = run_random_agent_on_all_tasks(n_episodes=50)
    assert mean(random_agent_scores[1]) > mean(random_agent_scores[2])
    assert mean(random_agent_scores[2]) > mean(random_agent_scores[3])
```

### OpenEnv Spec Compliance

```bash
# Run before every deployment
openenv validate openenv.yaml
openenv validate https://YOUR-SPACE.hf.space --live
```

### Pre-Submission Final Checklist

```
[ ] HF Space URL returns 200 on /health
[ ] POST /reset returns valid ExperimentObservation
[ ] POST /step returns valid StepResult with reward in [-1, 1]
[ ] openenv validate passes (local + live)
[ ] docker build && docker run succeeds locally
[ ] inference.py completes without error in <20 minutes
[ ] baseline_results.json contains scores for all 3 tasks
[ ] All grader scores in [0.0, 1.0]
[ ] Determinism: same seed always produces same score
[ ] README contains: description, action/obs spaces, task descriptions, setup, baseline scores
[ ] openenv.yaml contains required fields: name, version, tasks, action_space, observation_space
[ ] Space tagged with "openenv"
```

---

*Generated for the OpenEnv Round 1 Hackathon — A/B Test Causal Contamination Forensics Environment*  
*Framework: OpenEnv | Deployment target: Hugging Face Spaces | Deadline: April 1, 2025*