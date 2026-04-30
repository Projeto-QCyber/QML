from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from qml.schemas.agents_roles import Specialist
from qml.tools.model import RFModel
from qml.utils.api_call_models import _llm_default, _llm_leader
from qml.utils.runtime import crew_verbose


@CrewBase
class CyberPredict:
    """Um fluxo para detecção de ciberataques com votação de múltiplos agentes."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def cybersecurity_analyst_1(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_analyst_1'],
            tools=[RFModel()],
            llm=_llm_default(),
            verbose=crew_verbose(),
        )

    @agent
    def cybersecurity_analyst_2(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_analyst_2'],
            tools=[RFModel()],
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
        return Task(
            config=self.tasks_config['analyze_and_vote_1'],
            agent=self.cybersecurity_analyst_1()
        )

    @task
    def analyze_and_vote_2(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_2'],
            agent=self.cybersecurity_analyst_2()
        )

    @task
    def validate_results(self) -> Task:
        return Task(
            config=self.tasks_config['validate_results'],
            agent=self.cybersecurity_specialist(),
            output_file='src/qml/output/preliminary_prediction.txt',
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
        )
