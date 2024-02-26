"""Test Agent creation and execution basic functionality."""

from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.task import Task


def test_task_tool_reflect_agent_tools():
    """    Test the reflection of tools assigned to an agent for a task.

    This function sets up a fake tool and an agent, then creates a task with a specific description and agent. It then asserts that the tools assigned to the task match the expected tools.


    Raises:
        AssertionError: If the assigned tools do not match the expected tools.
    """

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        """        Fake tool

        This function represents a fake tool. It does not perform any actual functionality and serves as a placeholder.
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
    """    Test if task tool takes precedence over agent tools.

    This function sets up a test scenario to check if the task tool takes precedence over the agent tools. It creates a fake tool and a fake task tool using the langchain tools module. Then it creates an Agent and a Task using these tools and checks if the task tool takes precedence over the agent tools.

    Returns:
        This function does not return any value.
    """

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        """        Fake tool

        This function represents a fake tool and does not perform any actual functionality.
        It serves as a placeholder for a real tool.
        """

    @tool
    def fake_task_tool() -> None:
        """        Fake tool

        This function represents a fake task tool. It does not perform any actual task and is intended for testing or demonstration purposes.
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
    """    Test if the task prompt includes the expected output.

    This function sets up an Agent and a Task, then uses a mock to test if the task prompt includes the expected output.
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
    """    Test the callback function of a task.

    This function sets up a test scenario where a task is created with a researcher agent and a callback function. It then simulates the execution of the task and checks if the callback function is called with the expected output.
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
    """    Test the execution of a task by an agent.

    This function sets up a test scenario where a task is executed by an agent. It creates an instance of Agent and Task, and then uses a patch to mock the execute_task method of the Agent class. After executing the task, it asserts that the execute_task method was called with the expected arguments.
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
