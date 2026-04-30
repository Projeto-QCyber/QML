from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from qml.utils.api_call_models import _llm_coder_response


@CrewBase
class RemediationChatCrew:
    """Crew for operator-facing remediation chat and command suggestions."""

    agents_config = "config/agents_remediation.yaml"
    tasks_config = "config/tasks_remediation.yaml"

    @agent
    def remediation_operator_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["remediation_operator_assistant"],
            llm=_llm_coder_response(),
            verbose=True,
            allow_delegation=False,
            max_iter=2,
        )

    @task
    def remediation_chat(self) -> Task:
        return Task(
            config=self.tasks_config["remediation_chat"],
            agent=self.remediation_operator_assistant(),
            output_file="src/qml/output/remediation_chat.json",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.remediation_operator_assistant()],
            tasks=[self.remediation_chat()],
            process=Process.sequential,
            verbose=True,
        )
