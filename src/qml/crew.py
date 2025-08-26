from crewai import LLM, Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

import numpy as np
import pandas as pd

from schemas.agents_traditional import Specialist
from qml.tools.model import RFModel
from qml.utils.run_model import predict_model

@CrewBase
class CyberPredict:
    """Um fluxo para detecção de ciberataques com votação de múltiplos agentes."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @before_kickoff
    def prepare_inputs(self, input_path):
        df = pd.read_csv(input_path).sample(10, random_state=42)
        df_with_index = df.reset_index(drop=False)
        df_with_index.rename(columns={"index": "id"}, inplace=True)

        np_predictions, y_true, X = predict_model(df)
        baseline = pd.Series(np_predictions, index=df.index).astype(int)
        print(df_with_index)
        # Monte a lista de exemplos
        examples = []
        for i, row in df_with_index.iterrows():
            examples.append({
                "id": int(row["id"]),
                "y_true": int(y_true.loc[i]),
                "baseline_pred": int(baseline.loc[i]),
                "features": X.loc[i].to_dict(),
            })

        return {"examples": examples}

    @after_kickoff
    def process_output(self, output):
        output.raw += "\nProcessed after kickoff."
        return output

    @agent
    def cybersecurity_analyst_1(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_analyst_1'],
            #tools=[RFModel()],
            llm=LLM(
                model="ollama/qwen2.5:3b",
                base_url="http://localhost:11434"
            ),
            verbose=True
        )

    @agent
    def cybersecurity_analyst_2(self) -> Agent:
        return Agent(
            config=self.agents_config['cybersecurity_analyst_2'],
            #tools=[RFModel()],
            llm=LLM(
                model="ollama/qwen2.5:3b",
                base_url="http://localhost:11434"
            ),
            verbose=True,
        )

    @agent
    def cybersecurity_specialist(self) -> Agent:
      return Agent(
        config=self.agents_config['cybersecurity_specialist'],
        llm=LLM(
            model="ollama/qwen2.5:3b",
            base_url="http://localhost:11434"
        ),
        verbose=True
      )

    @task
    def analyze_and_vote_1(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_1']
        )

    @task
    def analyze_and_vote_2(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_and_vote_2']
        )
    
    @task
    def validate_results(self) -> Task:
      return Task(
        config=self.tasks_config['validate_results'],
        description="Especialista valida os votos e emite o resultado final.",
        agent=self.cybersecurity_specialist(),
        context=[self.analyze_and_vote_1(), self.analyze_and_vote_2()],
        output_file='./output/final_prediction.txt',
        output_pydantic=Specialist
      )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True)