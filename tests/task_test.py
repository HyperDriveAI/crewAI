"""Test Agent creation and execution basic functionality."""

from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.task import Task


def test_task_tool_reflect_agent_tools():
    """    Save the processed files map to a JSON file.

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

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
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

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        agent=researcher,
    )

    assert task.tools == [fake_tool]


def test_task_tool_takes_precedence_over_agent_tools():
    """    Save the processed files map to a JSON file.

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

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
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

    @tool
    def fake_task_tool() -> None:
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

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for an article, what makes them unique and interesting.",
        agent=researcher,
        tools=[fake_task_tool],
    )

    assert task.tools == [fake_task_tool]


def test_task_prompt_includes_expected_output():
    """    Save the processed files map to a JSON file.

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

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
        agent=researcher,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        task.execute()
        execute.assert_called_once_with(task=task._prompt(), context=None, tools=[])


def test_task_callback():
    """    Save the processed files map to a JSON file.

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

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task_completed = MagicMock(return_value="done")

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
        agent=researcher,
        callback=task_completed,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        task.execute()
        task_completed.assert_called_once_with(task.output)


def test_execute_with_agent():
    """    Save the processed files map to a JSON file.

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

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
    )

    with patch.object(Agent, "execute_task", return_value="ok") as execute:
        task.execute(agent=researcher)
        execute.assert_called_once_with(task=task._prompt(), context=None, tools=[])
