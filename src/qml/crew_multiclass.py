from __future__ import annotations

from typing import List
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent

# Your custom schema for the leader's final structured output
from qml.schemas.agents_roles import Specialist  # keep your Pydantic model
from qml.tools.model import RFModel              # your tool (RandomForest model wrapper)
from qml.utils.api_call_models import _llm_default, _llm_leader




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
    # 14 Specialist Agents (one per attack family)
    # The keys below must match your YAML in agents_mult.yaml
    # -------------------------------------------------------------------------
    @agent
    def cybersecurity_specialist_mitm(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_mitm'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_fingerprinting(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_fingerprinting'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ransomware(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ransomware'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_uploading(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_uploading'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_sql_injection(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_sql_injection'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_http(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_http'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_tcp(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_tcp'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_password(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_password'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_port_scanning(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_port_scanning'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_vulnerability_scanner(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_vulnerability_scanner'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_backdoor(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_backdoor'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_xss(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_xss'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_udp(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_udp'],
            tools=[self.rf_tool],
            llm=_llm_default(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def cybersecurity_specialist_ddos_icmp(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist_ddos_icmp'],
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
            verbose=True,
            allow_delegation=True,
            max_iter=1             
        )

    # -------------------------------------------------------------------------
    # 14 Analyze-and-vote tasks (one per specialist)
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

    # -------------------------------------------------------------------------
    # Final evaluation task (leader consolidates and explains rationale)
    # -------------------------------------------------------------------------
    @task
    def final_evaluation_task(self) -> Task:
        """
        The leader receives the outputs of all 14 specialist tasks through `context`
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
            ],
            output_file='src/qml/output/final_prediction.txt',
            output_pydantic=Specialist
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
        Orchestrates the 14 parallel(ish) specialist analyses plus the leader's arbitration.
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
            ],
            tasks=self.tasks,
            process=Process.hierarchical,             # leader-managed deliberation
            manager_agent=self.cybersecurity_team_leader(),  # explicit manager
            verbose=True
        )