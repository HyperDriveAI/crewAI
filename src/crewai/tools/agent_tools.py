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
        """        Save the processed files map to a JSON file.

        Function parameters should be documented in the ``Args`` section. The name
        of each parameter is required. The type and description of each parameter
        is optional, but should be included if not obvious.

        Args:
            dictionary (dict): The processed files map.

        Returns:
            bool: True if successful, False otherwise.
                The return type is optional and may be specified at the beginning of
                the ``Returns`` section followed by a colon.
                
                The ``Returns`` section may span multiple lines and paragraphs.
                Following lines should be indented to match the first line.
                
                The ``Returns`` section supports any reStructuredText formatting,
                including literal blocks::
                
                    {
                        'param1': param1,
                        'param2': param2
                    }
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
        """        Save the processed files map to a JSON file.

        Function parameters should be documented in the ``Args`` section. The name
        of each parameter is required. The type and description of each parameter
        is optional, but should be included if not obvious.

        Args:
            dictionary (dict): The processed files map.

        Returns:
            bool: True if successful, False otherwise.
                The return type is optional and may be specified at the beginning of
                the ``Returns`` section followed by a colon.
                
                The ``Returns`` section may span multiple lines and paragraphs.
                Following lines should be indented to match the first line.
                
                The ``Returns`` section supports any reStructuredText formatting,
                including literal blocks::
                
                    {
                        'param1': param1,
                        'param2': param2
                    }
        """
        return self._execute(command)

    def ask_question(self, command):
        """        Save the processed files map to a JSON file.

        Function parameters should be documented in the ``Args`` section. The name
        of each parameter is required. The type and description of each parameter
        is optional, but should be included if not obvious.

        Args:
            dictionary (dict): The processed files map.

        Returns:
            bool: True if successful, False otherwise.
                The return type is optional and may be specified at the beginning of
                the ``Returns`` section followed by a colon.
                
                The ``Returns`` section may span multiple lines and paragraphs.
                Following lines should be indented to match the first line.
                
                The ``Returns`` section supports any reStructuredText formatting,
                including literal blocks::
                
                    {
                        'param1': param1,
                        'param2': param2
                    }
        """
        return self._execute(command)

    def _execute(self, command):
        """        Save the processed files map to a JSON file.

        Function parameters should be documented in the ``Args`` section. The name
        of each parameter is required. The type and description of each parameter
        is optional, but should be included if not obvious.

        Args:
            dictionary (dict): The processed files map.

        Returns:
            bool: True if successful, False otherwise.
                The return type is optional and may be specified at the beginning of
                the ``Returns`` section followed by a colon.
                
                The ``Returns`` section may span multiple lines and paragraphs.
                Following lines should be indented to match the first line.
                
                The ``Returns`` section supports any reStructuredText formatting,
                including literal blocks::
                
                    {
                        'param1': param1,
                        'param2': param2
                    }
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
