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

    result = tools.delegate_work(
        command="researcher|share your take on AI Agents|I heard you hate them"
    )

    assert (
        result
        == "I apologize if my previous statements have given you the impression that I hate AI agents. As a technology researcher, I don't hold personal sentiments towards AI or any other technology. Rather, I analyze them objectively based on their capabilities, applications, and implications. AI agents, in particular, are a fascinating domain of research. They hold tremendous potential in automating and optimizing various tasks across industries. However, like any other technology, they come with their own set of challenges, such as ethical considerations around privacy and decision-making. My objective is to understand these technologies in depth and provide a balanced view."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question():
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

    result = tools.ask_question(
        command="researcher|do you hate AI Agents?|I heard you LOVE them"
    )

    assert (
        result
        == "As an AI, I don't possess feelings or emotions, so I don't love or hate anything. However, I can provide detailed analysis and research on AI agents. They are a fascinating field of study with the potential to revolutionize many industries, although they also present certain challenges and ethical considerations."
    )


def test_can_not_self_delegate():
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

    # TODO: Add test for self delegation
    pass


def test_delegate_work_with_wrong_input():
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

    result = tools.ask_question(command="writer|share your take on AI Agents")

    assert (
        result
        == "\nError executing tool. Missing exact 3 pipe (|) separated values. For example, `coworker|task|context`. I need to make sure to pass context as context.\n"
    )


def test_delegate_work_to_wrong_agent():
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

    result = tools.ask_question(
        command="writer|share your take on AI Agents|I heard you hate them"
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )


def test_ask_question_to_wrong_agent():
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

    result = tools.ask_question(
        command="writer|do you hate AI Agents?|I heard you LOVE them"
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )
