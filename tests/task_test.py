"""Test Agent creation and execution basic functionality."""

from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.task import Task


def test_task_tool_reflect_agent_tools():
    """Test the creation of a task with a tool reflecting an agent's tools.

    This function demonstrates how to create a researcher agent, define a
    tool, and assign it to a task. It then asserts that the task contains
    the specified tool.
    """

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        "Fake tool"

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
    """Ensure that a task tool takes precedence over agent tools in the context
    of an Agent.

    This function sets up an Agent with specific roles, goals, and tools. It
    then creates a Task for the Agent to perform, specifying different
    tools. The function asserts that the task's tools list is updated to use
    the task-specific tool rather than the agent-specific tool.
    """

    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        "Fake tool"

    @tool
    def fake_task_tool() -> None:
        "Fake tool"

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
    """Test that the task prompt includes the expected output.

    This function creates a `Researcher` agent and assigns it to a `Task`.
    It then patches the `execute_task` method of the `Agent` class to return
    "ok" when called. The function asserts that calling `task.execute()`
    results in the `execute_task` method being called once with the expected
    arguments.
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
    """Test the callback functionality of a Task object when executed.

    This function sets up a scenario where a `Task` is created with an
    `Agent` and a `callback` method. It then simulates executing the task
    and verifies that the callback method is called once with the expected
    output from the task.  The function uses mocking to control the behavior
    of the `execute_task` method of the `Agent` class, ensuring it returns a
    predefined value, and checks if the callback method is indeed invoked
    with the correct arguments.
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
    """Tests the execution of a Task with an Agent.

    This function creates an instance of the Agent and Task classes, then
    executes the task using the agent. It uses the `patch.object` from the
    `unittest.mock` module to mock the `execute_task` method of the Agent
    class to return "ok". After executing the task, it asserts that the
    `execute_task` method was called once with the correct arguments.
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
