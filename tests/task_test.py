"""Test Agent creation and execution basic functionality."""

from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.task import Task


def test_task_tool_reflect_agent_tools():
    """Test the task tool reflection for agent tools.

    This function sets up a test scenario for an agent that utilizes a fake
    tool. It creates an instance of the `Agent` class with specific
    attributes such as role, goal, and backstory. A `Task` is then
    instantiated with a description that prompts the agent to generate ideas
    for an article. The test concludes by asserting that the tools
    associated with the task match the expected fake tool.
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
    """Test that task tools take precedence over agent tools.

    This function verifies that when a task is created with specific tools,
    those tools override any tools that may be associated with the agent. It
    sets up a fake tool and a fake task tool, then creates an agent with the
    fake tool and a task with the fake task tool. The assertion checks that
    the tools associated with the task are indeed the task tools, confirming
    the expected behavior of precedence.
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

    This function sets up a test case for the `Task` class, specifically
    verifying that the prompt generated for a task executed by an `Agent`
    instance matches the expected output. It creates a `Researcher` agent
    with a specific role and goal, and then defines a task that requires
    generating a list of interesting article ideas. The test uses mocking to
    intercept the call to the `execute_task` method of the `Agent`, allowing
    verification that it is called with the correct parameters.
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
    """Test the task callback functionality of the Task class.

    This function sets up a test scenario for the Task class, specifically
    focusing on the callback mechanism after task execution. It creates a
    Researcher agent with specific attributes and a task that requires
    generating a list of interesting article ideas. The test verifies that
    the callback is called with the correct output after the task is
    executed.
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
    """Test the execution of a task with an agent.

    This function sets up a test scenario where an agent, specifically a
    researcher, is tasked with generating a list of interesting ideas for an
    article. The agent's role, goal, and backstory are defined to simulate a
    realistic research environment. The task is then executed, and the
    function verifies that the agent's `execute_task` method is called with
    the correct parameters.
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
