from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from qml.utils.api_call_models import _llm_coder_response, _llm_leader
from qml.utils.runtime import crew_verbose

@CrewBase
class IncidentResponseCrew:
    """Crew para gerar planos de resposta a incidentes."""

    agents_config = 'config/agents_response.yaml'
    tasks_config = 'config/tasks_response.yaml'

    @agent
    def incident_responder(self) -> Agent:
        """
        Define o agente planejador de resposta a incidentes.
        """
        return Agent(
            config=self.agents_config['incident_responder'],
            llm=_llm_coder_response(),
            verbose=crew_verbose(),
            allow_delegation=False,
            max_iter=2,
        )

    @task
    def generate_response_plan(self) -> Task:
        """
        Define a tarefa para gerar o plano de resposta.
        """
        return Task(
            config=self.tasks_config['generate_response_plan'],
            agent=self.incident_responder(),
            output_file='src/qml/output/incident_response_plan.md',
        )

    @crew
    def crew(self) -> Crew:
        """
        Monta e retorna o crew com o agente e a tarefa.
        """
        return Crew(
            agents=[self.incident_responder()],
            tasks=[self.generate_response_plan()],
            process=Process.sequential,
            verbose=crew_verbose(),
        )
