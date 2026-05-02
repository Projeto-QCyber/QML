from __future__ import annotations

from typing import List
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent

# Your custom schemas
from qml.schemas.agents_roles import Specialist, LeaderDecision  # specialist votes and leader decision
from qml.tools.model import RFModel              # your tool (RandomForest model wrapper)
from qml.utils.api_call_models import _llm_default, _llm_leader
from qml.utils.runtime import crew_verbose


CLASS_CREW_METHODS = {
    0: ("cybersecurity_specialist_backdoor", "analyze_and_vote_backdoor"),
    1: ("cybersecurity_specialist_ddos_http", "analyze_and_vote_ddos_http"),
    2: ("cybersecurity_specialist_ddos_icmp", "analyze_and_vote_ddos_icmp"),
    3: ("cybersecurity_specialist_ddos_tcp", "analyze_and_vote_ddos_tcp"),
    4: ("cybersecurity_specialist_ddos_udp", "analyze_and_vote_ddos_udp"),
    5: ("cybersecurity_specialist_fingerprinting", "analyze_and_vote_fingerprinting"),
    6: ("cybersecurity_specialist_mitm", "analyze_and_vote_mitm"),
    7: ("cybersecurity_specialist_password", "analyze_and_vote_password"),
    8: ("cybersecurity_specialist_port_scanning", "analyze_and_vote_port_scanning"),
    9: ("cybersecurity_specialist_ransomware", "analyze_and_vote_ransomware"),
    10: ("cybersecurity_specialist_sql_injection", "analyze_and_vote_sql_injection"),
    11: ("cybersecurity_specialist_uploading", "analyze_and_vote_uploading"),
    12: ("cybersecurity_specialist_vulnerability_scanner", "analyze_and_vote_vulnerability_scanner"),
    13: ("cybersecurity_specialist_xss", "analyze_and_vote_xss"),
    14: ("cybersecurity_specialist_outras", "analyze_and_vote_outras"),
}


def run_shap_explanation(model, sample_df: pd.DataFrame):
    import shap
    import time

    start_time = time.time()
    print("\n[SHAP] Starting SHAP analysis for the sample...")

    try:
        if hasattr(model, "feature_names_in_"):
            sample_df = sample_df.reindex(columns=model.feature_names_in_, fill_value=0)

        print("\n[SHAP] Starting SHAP analysis for the sample...")

        explainer = shap.TreeExplainer(model)

        print("\n[SHAP] Explainer created.")

        shap_values = explainer.shap_values(sample_df.values)

        print("\n[SHAP] SHAP values computed.")
        
        feature_names = sample_df.columns
        feature_shap_values = dict(zip(feature_names, shap_values))
        sorted_features = sorted(feature_shap_values.items(), key=lambda item: abs(item[1]), reverse=True)

        print(f"\n[SHAP] Features sorted. feature_names: {feature_names}, feature_shap_values: {feature_shap_values}, sorted_features: {sorted_features}")

        explanation = "The top 2 features influencing this prediction were:\n"

        print(f"\n[SHAP] Explanation: {explanation}")

        for feature, shap_value in sorted_features[:2]:
            effect = "positively" if shap_value > 0 else "negatively"
            explanation += f"- The feature '{feature}' had a strong {effect} impact on the decision.\n"
        
        elapsed_time = time.time() - start_time
        print(f"[SHAP] ✅ Analysis finished in {elapsed_time:.2f} seconds.")
        return explanation

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_message = f"[SHAP] ❌ ERROR: Failed to generate SHAP explanation in {elapsed_time:.2f}s. Reason: {e}"
        print(error_message)
        return "SHAP analysis could not be performed due to an internal error."



@CrewBase
class CyberPredictMult:
    """
    Multi-agent flow for cyberattack detection with parallel specialist votes
    and a leader-driven discussion to converge on the final label.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents_mult.yaml'
    tasks_config = 'config/tasks_mult.yaml'

    def __init__(self):
        """Initializes a single, shared RFModel tool instance for all agents."""
        self.rf_tool = RFModel(classification="multiclass")

    # -------------------------------------------------------------------------
    # 15 Specialist Agents (one per attack family)
    # The keys below must match your YAML in agents_mult.yaml
    # -------------------------------------------------------------------------
    @agent
    def cybersecurity_specialist_mitm(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_mitm'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_fingerprinting(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_fingerprinting'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ransomware(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ransomware'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_uploading(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_uploading'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_sql_injection(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_sql_injection'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_http(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_http'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_tcp(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_tcp'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_password(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_password'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_port_scanning(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_port_scanning'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_vulnerability_scanner(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_vulnerability_scanner'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_backdoor(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_backdoor'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_xss(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_xss'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_udp(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_udp'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_icmp(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_icmp'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=crew_verbose(),
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_outras(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_outras'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    # -------------------------------------------------------------------------
    # Team Leader (moderator / final arbiter)
    # -------------------------------------------------------------------------
    @agent
    def cybersecurity_team_leader(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_team_leader'],
            llm=_llm_leader(),
            verbose=crew_verbose(),
            allow_delegation=True,
            max_iter=1             
        )

    # -------------------------------------------------------------------------
    # 15 Analyze-and-vote tasks (one per specialist)
    # The keys below must match your YAML in tasks_mult.yaml
    # -------------------------------------------------------------------------
    @task
    def analyze_and_vote_mitm(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_mitm'],
            agent=self.cybersecurity_specialist_mitm()
        )

    @task
    def analyze_and_vote_fingerprinting(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_fingerprinting'],
            agent=self.cybersecurity_specialist_fingerprinting()
        )

    @task
    def analyze_and_vote_ransomware(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_ransomware'],
            agent=self.cybersecurity_specialist_ransomware()
        )

    @task
    def analyze_and_vote_uploading(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_uploading'],
            agent=self.cybersecurity_specialist_uploading()
        )

    @task
    def analyze_and_vote_sql_injection(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_sql_injection'],
            agent=self.cybersecurity_specialist_sql_injection()
        )

    @task
    def analyze_and_vote_ddos_http(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_ddos_http'],
            agent=self.cybersecurity_specialist_ddos_http()
        )

    @task
    def analyze_and_vote_ddos_tcp(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_ddos_tcp'],
            agent=self.cybersecurity_specialist_ddos_tcp()
        )

    @task
    def analyze_and_vote_password(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_password'],
            agent=self.cybersecurity_specialist_password()
        )

    @task
    def analyze_and_vote_port_scanning(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_port_scanning'],
            agent=self.cybersecurity_specialist_port_scanning()
        )

    @task
    def analyze_and_vote_vulnerability_scanner(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_vulnerability_scanner'],
            agent=self.cybersecurity_specialist_vulnerability_scanner()
        )

    @task
    def analyze_and_vote_backdoor(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_backdoor'],
            agent=self.cybersecurity_specialist_backdoor()
        )

    @task
    def analyze_and_vote_xss(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_xss'],
            agent=self.cybersecurity_specialist_xss()
        )

    @task
    def analyze_and_vote_ddos_udp(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_ddos_udp'],
            agent=self.cybersecurity_specialist_ddos_udp()
        )

    @task
    def analyze_and_vote_ddos_icmp(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_ddos_icmp'],
            agent=self.cybersecurity_specialist_ddos_icmp()
        )

    @task
    def analyze_and_vote_outras(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_outras'],
            agent=self.cybersecurity_specialist_outras()
        )

    # -------------------------------------------------------------------------
    # Final evaluation task (leader consolidates and explains rationale)
    # -------------------------------------------------------------------------
    @task
    def final_evaluation_task(self) -> Task:
        """
        The leader receives the outputs of all 15 specialist tasks through `context`
        and must return a final per-sample label with short justification.
        """
        return Task(
            config=self.tasks_config['final_evaluation_task'],
            agent=self.cybersecurity_team_leader(),
            context=[
                self.analyze_and_vote_mitm(),
                self.analyze_and_vote_fingerprinting(),
                self.analyze_and_vote_ransomware(),
                self.analyze_and_vote_uploading(),
                self.analyze_and_vote_sql_injection(),
                self.analyze_and_vote_ddos_http(),
                self.analyze_and_vote_ddos_tcp(),
                self.analyze_and_vote_password(),
                self.analyze_and_vote_port_scanning(),
                self.analyze_and_vote_vulnerability_scanner(),
                self.analyze_and_vote_backdoor(),
                self.analyze_and_vote_xss(),
                self.analyze_and_vote_ddos_udp(),
                self.analyze_and_vote_ddos_icmp(),
                self.analyze_and_vote_outras(),
            ],
            expected_output=(
                "Return STRICT JSON only, exactly in this schema: "
                "{\"predictions\":[<int>], \"report\":\"<short explanation>\"}. "
                "Rules: Do NOT include any text before/after the JSON. Do NOT use code fences. "
                "Do NOT include <think> hidden thoughts or LaTeX (e.g., \\boxed). "
                "Use only integers for predictions (single element list), e.g., [10]. "
                "If uncertain, return {\"predictions\":[99], \"report\":\"Normal\"}."
            ),
            output_pydantic=LeaderDecision,
            output_file='src/qml/output/final_prediction.json',
        )

    def final_evaluation_task_for_context(self, context_tasks: list[Task]) -> Task:
        return Task(
            config=self.tasks_config['final_evaluation_task'],
            agent=self.cybersecurity_team_leader(),
            context=context_tasks,
            expected_output=(
                "Return STRICT JSON only, exactly in this schema: "
                "{\"predictions\":[<int>], \"report\":\"<short explanation>\"}. "
                "Rules: Do NOT include any text before/after the JSON. Do NOT use code fences. "
                "Do NOT include <think> hidden thoughts or LaTeX (e.g., \\boxed). "
                "Use only integers for predictions (single element list), e.g., [10]. "
                "If uncertain, return {\"predictions\":[99], \"report\":\"Normal\"}."
            ),
            output_pydantic=LeaderDecision,
            output_file='src/qml/output/final_prediction.json',
        )

    def crew_for_class_ids(self, class_ids: list[int]) -> Crew:
        selected_ids = []
        for class_id in class_ids:
            normalized_id = int(class_id)
            if normalized_id in CLASS_CREW_METHODS and normalized_id not in selected_ids:
                selected_ids.append(normalized_id)

        if not selected_ids:
            selected_ids = list(CLASS_CREW_METHODS)

        selected_agents = []
        selected_tasks = []
        for class_id in selected_ids:
            agent_method_name, task_method_name = CLASS_CREW_METHODS[class_id]
            selected_agents.append(getattr(self, agent_method_name)())
            selected_tasks.append(getattr(self, task_method_name)())

        final_task = self.final_evaluation_task_for_context(selected_tasks)
        return Crew(
            agents=[*selected_agents, self.cybersecurity_team_leader()],
            tasks=[*selected_tasks, final_task],
            process=Process.sequential,
            verbose=crew_verbose(),
        )

    # -------------------------------------------------------------------------
    # Crew definition
    # Using hierarchical process allows the leader to coordinate a "discussion".
    # Each analyze-and-vote task runs independently; CrewAI will orchestrate,
    # and the leader consumes their outputs for the final decision.
    # -------------------------------------------------------------------------
    @crew
    def crew(self) -> Crew:
        """
        Orchestrates the 15 parallel(ish) specialist analyses plus the leader's arbitration.
        Note: CrewAI executes tasks internally; with the hierarchical process,
        the team leader acts as a manager that 'discusses' and converges decisions.
        """
        return Crew(
            agents=[
                self.cybersecurity_specialist_backdoor(),
                self.cybersecurity_specialist_fingerprinting(),
                self.cybersecurity_specialist_ransomware(),
                self.cybersecurity_specialist_uploading(),
                self.cybersecurity_specialist_sql_injection(),
                self.cybersecurity_specialist_ddos_http(),
                self.cybersecurity_specialist_ddos_tcp(),
                self.cybersecurity_specialist_password(),
                self.cybersecurity_specialist_port_scanning(),
                self.cybersecurity_specialist_vulnerability_scanner(),
                self.cybersecurity_specialist_xss(),
                self.cybersecurity_specialist_ddos_udp(),
                self.cybersecurity_specialist_ddos_icmp(),
                self.cybersecurity_specialist_outras(),
            ],
            tasks=self.tasks,
            process=Process.hierarchical,             # leader-managed deliberation
            manager_agent=self.cybersecurity_team_leader(),  # explicit manager
            verbose=crew_verbose()
        )
