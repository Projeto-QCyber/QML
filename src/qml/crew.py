import os
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from qml.tools.model import RFModel
from qml.schemas.agents_roles import Specialist
from qml.tools.quantum_model import QuantumModel
from qml.utils.api_call_models import _llm_default, _llm_leader
from qml.utils.runtime import crew_verbose


@CrewBase
class CyberPredict:
    """Um fluxo para detecção de ciberataques com votação de múltiplos agentes."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def _binary_tool_provider(self) -> str:
        provider = os.getenv("QML_BINARY_TOOL_PROVIDER", "both").strip().lower()
        print(f"provider: {provider}")
        if provider not in {"both", "rf", "quantum"}:
            raise ValueError("QML_BINARY_TOOL_PROVIDER must be 'both', 'rf', or 'quantum'.")
        return provider

    def _binary_tools(self):
        provider = self._binary_tool_provider()
        if provider == "quantum":
            return [QuantumModel()]
        if provider == "rf":
            return [RFModel()]
        return [RFModel(), QuantumModel()]

    def _apply_binary_tool_instruction(self, task_obj: Task) -> Task:
        provider = self._binary_tool_provider()
        if provider == "quantum":
            task_obj.description += (
                "\n\nTEST MODE: Use ONLY the `quantum_model` tool for this task. "
                "Do not call the `Model` tool."
            )
        elif provider == "rf":
            task_obj.description += (
                "\n\nTEST MODE: Use ONLY the `Model` tool for this task. "
                "Do not call the `quantum_model` tool."
            )
        return task_obj

    @agent
    def cybersecurity_analyst_1(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_analyst_1'],
            tools=self._binary_tools(),
            llm=_llm_default(),
            verbose=crew_verbose(),
        )

    @agent
    def cybersecurity_analyst_2(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_analyst_2'],
            tools=self._binary_tools(),
            llm=_llm_default(),
            verbose=crew_verbose(),
        )

    @agent
    def cybersecurity_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_specialist'],
            llm=_llm_leader(),
            verbose=crew_verbose(),
        )

    @task
    def analyze_and_vote_1(self) -> Task:
        return self._apply_binary_tool_instruction(Task(
            config=self.tasks_config['analyze_and_vote_1'],
            agent=self.cybersecurity_analyst_1()
        ))

    @task
    def analyze_and_vote_2(self) -> Task:
        return self._apply_binary_tool_instruction(Task(
            config=self.tasks_config['analyze_and_vote_2'],
            agent=self.cybersecurity_analyst_2()
        ))

    @task
    def validate_results(self) -> Task:
        return Task(
            config=self.tasks_config['validate_results'],
            agent=self.cybersecurity_specialist(),
            output_file='src/qml/output/preliminary_prediction.json',
        )

    @crew
    def crew(self) -> Crew:
        task1 = self.analyze_and_vote_1()
        task2 = self.analyze_and_vote_2()
        task3 = self.validate_results()

        # Garante que as saídas de task1 e task2 estejam disponíveis para task3
        task3.context = [task1, task2]

        return Crew(
            agents=self.agents,
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=crew_verbose(),
            max_iter=2,
        )
