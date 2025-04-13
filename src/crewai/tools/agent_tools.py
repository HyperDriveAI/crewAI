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
        """Generate a list of tools available for delegation and communication.

        This function creates and returns a list of tools, each represented by
        an instance of `Tool`. Each tool is initialized with a specific function
        and a description. The descriptions are dynamically formatted to include
        the roles of the co-workers available in the current context.

        Returns:
            list: A list containing two tools - one for delegating work and another for
                asking questions.
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
        """Delegate a specific task to a coworker.

        Args:
            command (str): The task or command to be delegated to a coworker.
        """
        return self._execute(command)

    def ask_question(self, command):
        """Ask a question, opinion, or request information from a coworker.

        This method is useful for initiating conversations or gathering insights
        by sending a message to a coworker and expecting a response.

        Args:
            command (str): The message or question to be sent to the coworker.

        Returns:
            str: The response received from the coworker.
        """
        return self._execute(command)

    def _execute(self, command):
        """Execute the given command.

        This function processes a command string by splitting it into its
        constituent parts: 'agent', 'task', and 'context'. It then checks if all
        parts are present. If any part is missing, it returns an error message.
        Otherwise, it filters available agents based on the specified role and
        executes the task with the given context. If no matching agent is found,
        it returns a different error message.

        Args:
            command (str): A string containing the 'agent', 'task', and 'context' separated by '|'.

        Returns:
            Any: The result of executing the task if successful; otherwise, an error
                message.
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
