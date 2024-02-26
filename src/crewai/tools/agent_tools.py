from typing import List

from langchain.tools import Tool
from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.utilities import I18N


class AgentTools(BaseModel):
    """Default tools around agent delegation"""

    agents: List[Agent] = Field(description="List of agents in this crew.")
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")

    def tools(self):
        """        Generate tools for delegating work and asking questions to co-workers.

        This function returns a list of Tool objects, each representing a specific tool for delegating work or asking questions
        to co-workers. The Tool objects are created using the Tool.from_function method, with each tool having a name and a
        description that is generated based on the available co-workers.

        Returns:
            list: A list of Tool objects representing tools for delegating work and asking questions to co-workers.
        """

        return [
            Tool.from_function(
                func=self.delegate_work,
                name="Delegate work to co-worker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers=", ".join([agent.role for agent in self.agents])
                ),
            ),
            Tool.from_function(
                func=self.ask_question,
                name="Ask question to co-worker",
                description=self.i18n.tools("ask_question").format(
                    coworkers=", ".join([agent.role for agent in self.agents])
                ),
            ),
        ]

    def delegate_work(self, command):
        """        Useful to delegate a specific task to a coworker.

        This function is used to delegate a specific task to a coworker. It takes a command as input and executes it using the
        _execute method.

        Args:
            command: The command to be delegated.

        Returns:
            The result of executing the command.
        """
        return self._execute(command)

    def ask_question(self, command):
        """        Useful to ask a question, opinion or take from a coworker.

        This function is used to ask a question, opinion, or take from a coworker. It takes a command as input and
        executes it using the _execute method of the current object.

        Args:
            self: The current object.
            command: The command to be executed.

        Returns:
            The result of executing the command using the _execute method.
        """
        return self._execute(command)

    def _execute(self, command):
        """        Execute the command.

        This method takes a command as input and executes it. The command should be in the format 'agent|task|context'.
        It first tries to split the command into agent, task, and context. If the command does not follow the correct format,
        it raises a ValueError. If any of the agent, task, or context is missing, it raises an error. If the specified agent
        does not exist in the available agents, it raises an error. Finally, it executes the task using the specified agent
        and returns the result.

        Args:
            command (str): The command to be executed in the format 'agent|task|context'.

        Returns:
            str: The result of executing the task using the specified agent.

        Raises:
            ValueError: If the command does not follow the correct format.
        """
        try:
            agent, task, context = command.split("|")
        except ValueError:
            return self.i18n.errors("agent_tool_missing_param")

        if not agent or not task or not context:
            return self.i18n.errors("agent_tool_missing_param")

        agent = [
            available_agent
            for available_agent in self.agents
            if available_agent.role == agent
        ]

        if not agent:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers=", ".join([agent.role for agent in self.agents])
            )

        agent = agent[0]
        return agent.execute_task(task, context)
