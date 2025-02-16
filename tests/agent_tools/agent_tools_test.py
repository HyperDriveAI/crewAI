"""Test Agent creation and execution basic functionality."""

import pytest

from crewai.agent import Agent
from crewai.tools.agent_tools import AgentTools

researcher = Agent(
    role="researcher",
    goal="make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology",
    allow_delegation=False,
)
tools = AgentTools(agents=[researcher])


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work():
    """Test the delegate_work function from the tools module.

    This function tests the behavior of the delegate_work method by
    providing a specific command string. It asserts that the output from the
    method matches the expected response regarding AI agents. The test
    ensures that the delegate_work function can generate a coherent and
    contextually appropriate response based on the input command.
    """

    result = tools.delegate_work(
        command="researcher|share your take on AI Agents|I heard you hate them"
    )

    assert (
        result
        == "I apologize if my previous statements have given you the impression that I hate AI agents. As a technology researcher, I don't hold personal sentiments towards AI or any other technology. Rather, I analyze them objectively based on their capabilities, applications, and implications. AI agents, in particular, are a fascinating domain of research. They hold tremendous potential in automating and optimizing various tasks across industries. However, like any other technology, they come with their own set of challenges, such as ethical considerations around privacy and decision-making. My objective is to understand these technologies in depth and provide a balanced view."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question():
    """Test the ask_question function from the tools module.

    This function tests the behavior of the ask_question method by providing
    a specific command and asserting that the returned response matches the
    expected output. The command simulates a question about AI agents, and
    the expected response is a neutral statement about the nature of AI and
    its capabilities.
    """

    result = tools.ask_question(
        command="researcher|do you hate AI Agents?|I heard you LOVE them"
    )

    assert (
        result
        == "As an AI, I don't possess feelings or emotions, so I don't love or hate anything. However, I can provide detailed analysis and research on AI agents. They are a fascinating field of study with the potential to revolutionize many industries, although they also present certain challenges and ethical considerations."
    )


def test_can_not_self_delegate():
    # TODO: Add test for self delegation
    pass


def test_delegate_work_with_wrong_input():
    """Test the behavior of the delegate work function with incorrect input.

    This function tests the `ask_question` method from the `tools` module by
    providing an improperly formatted command string. The test asserts that
    the output matches the expected error message, which indicates that the
    input must contain exactly three pipe-separated values. This ensures
    that the function handles invalid input gracefully and provides clear
    feedback to the user.
    """

    result = tools.ask_question(command="writer|share your take on AI Agents")

    assert (
        result
        == "\nError executing tool. Missing exact 3 pipe (|) separated values. For example, `coworker|task|context`. I need to make sure to pass context as context.\n"
    )


def test_delegate_work_to_wrong_agent():
    """Test delegating work to an incorrect agent.

    This function tests the behavior of the system when a question is asked
    to an agent that is not designated for the task. It simulates asking a
    question to a writer agent while the expected agent is a researcher. The
    test asserts that the correct error message is returned when the wrong
    agent is invoked.
    """

    result = tools.ask_question(
        command="writer|share your take on AI Agents|I heard you hate them"
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )


def test_ask_question_to_wrong_agent():
    """Test asking a question to an incorrect agent.

    This function tests the behavior of the `ask_question` method when it is
    invoked with a command directed at an agent that is not available.
    Specifically, it checks that the appropriate error message is returned
    when the command specifies an agent that is not in the list of valid
    options.  The test asserts that the result of the `ask_question` call
    matches the expected error message, indicating that the system correctly
    handles requests to non-existent agents.
    """

    result = tools.ask_question(
        command="writer|do you hate AI Agents?|I heard you LOVE them"
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )
