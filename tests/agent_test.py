"""Test Agent creation and execution basic functionality."""

from unittest.mock import patch

import pytest
from langchain.tools import tool
from langchain_openai import ChatOpenAI as OpenAI

from crewai import Agent, Crew, Task
from crewai.agents.cache import CacheHandler
from crewai.agents.executor import CrewAgentExecutor
from crewai.utilities import RPMController


def test_agent_creation():
    """    Test the creation of an Agent object.

    This function tests the creation of an Agent object with the specified role, goal, and backstory. It asserts that the
    attributes of the created Agent object match the provided values.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []


def test_agent_default_values():
    """    Test the default values of an Agent.

    This function creates an Agent object with specified role, goal, and backstory, and then asserts that the default values of its attributes are as expected.


    Raises:
        AssertionError: If any of the assertions fail.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0.7
    assert agent.llm.verbose == False
    assert agent.allow_delegation == True


def test_custom_llm():
    """    Test the custom long-term language model (LLM) for an agent.

    This function initializes an agent with a custom long-term language model (LLM) using the OpenAI platform.
    It then asserts that the LLM is an instance of the OpenAI class and checks specific attributes of the LLM.


    Raises:
        AssertionError: If the assertions fail.
    """

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_without_memory():
    """    Test the Agent class without memory.

    This function creates two instances of the Agent class, one with memory set to False and one with memory set to True.
    It then executes a task for the agent without memory and asserts the result and memory state of both agents.


    Raises:
        AssertionError: If the result is not as expected or if the memory state is not as expected.
    """

    no_memory_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        memory=False,
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    memory_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        memory=True,
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    result = no_memory_agent.execute_task("How much is 1 + 1?")

    assert result == "1 + 1 equals 2."
    assert no_memory_agent.agent_executor.memory is None
    assert memory_agent.agent_executor.memory is not None


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution():
    """    Test the execution of an agent's task.

    This function creates an agent with the specified role, goal, backstory, and delegation settings. It then executes a task and asserts that the output matches the expected result.


    Raises:
        AssertionError: If the output does not match the expected result.
    """

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    output = agent.execute_task("How much is 1 + 1?")
    assert output == "2"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_tools():
    """    Test the execution of an agent with tools.

    This function tests the execution of an agent with tools. It creates a multiplier tool
    that takes a comma-separated list of two numbers and multiplies them together. Then it
    creates an agent with a specified role, goal, backstory, and the multiplier tool. The
    agent is then tasked with executing the question "What is 3 times 4", and the output is
    asserted to be "12".
    """

    @tool
    def multiplier(numbers) -> float:
        """        Useful for when you need to multiply two numbers together.

        The input to this tool should be a comma separated list of numbers of
        length two, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2.

        Args:
            numbers (str): A comma separated string of two numbers.

        Returns:
            float: The result of multiplying the two input numbers.
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
    )

    output = agent.execute_task("What is 3 times 4")
    assert output == "12"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_logging_tool_usage():
    """    Test the usage of logging tool.

    This function tests the usage of a logging tool by creating an Agent with a multiplier tool and
    executing a task to multiply two numbers. It checks if the output and the last used tool match the expected values.


    Raises:
        AssertionError: If the output or the last used tool does not match the expected values.
    """

    @tool
    def multiplier(numbers) -> float:
        """        Useful for when you need to multiply two numbers together.

        The input to this tool should be a comma separated list of numbers of
        length two, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2.

        Args:
            numbers (str): A comma separated list of two numbers to be multiplied.

        Returns:
            float: The result of multiplying the two input numbers together.
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        verbose=True,
    )

    assert agent.tools_handler.last_used_tool == {}
    output = agent.execute_task("What is 3 times 5?")
    tool_usage = {
        "tool": "multiplier",
        "input": "3,5",
    }

    assert output == "3 times 5 is 15."
    assert agent.tools_handler.last_used_tool == tool_usage


@pytest.mark.vcr(filter_headers=["authorization"])
def test_cache_hitting():
    """    Test the cache handling functionality of the Agent class.

    This function tests the cache handling functionality of the Agent class by creating a multiplier tool
    and using it to perform multiplication operations. It then checks if the cache is updated correctly and
    if the agent is able to retrieve results from the cache.

    Returns:
        str: The result of the multiplication operation.

    Raises:
        AssertionError: If the cache handling or retrieval fails.
    """

    @tool
    def multiplier(numbers) -> float:
        """        Useful for when you need to multiply two numbers together.

        The input to this tool should be a comma separated list of numbers of
        length two and ONLY TWO, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2.

        Args:
            numbers (str): A comma separated list of two numbers to be multiplied.

        Returns:
            float: The result of multiplying the two input numbers.

        Raises:
            ValueError: If the input does not follow the specified format.
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    cache_handler = CacheHandler()

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        cache_handler=cache_handler,
        verbose=True,
    )

    output = agent.execute_task("What is 2 times 6 times 3?")
    output = agent.execute_task("What is 3 times 3?")
    assert cache_handler._cache == {
        "multiplier-12,3": "36",
        "multiplier-2,6": "12",
        "multiplier-3,3": "9",
    }

    output = agent.execute_task("What is 2 times 6 times 3? Return only the number")
    assert output == "36"

    with patch.object(CacheHandler, "read") as read:
        read.return_value = "0"
        output = agent.execute_task("What is 2 times 6?")
        assert output == "0"
        read.assert_called_with("multiplier", "2,6")


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_specific_tools():
    """    Test the execution of an agent with specific tools.

    This function tests the execution of an agent with specific tools. It creates a multiplier tool
    that takes a comma-separated list of two numbers as input and returns their product. Then, it
    creates an agent with specified role, goal, backstory, and delegation settings. The agent is
    then tasked with a question that involves using the multiplier tool, and the output is checked
    for correctness.

    Returns:
        This function does not return any value.
    """

    @tool
    def multiplier(numbers) -> float:
        """        Useful for when you need to multiply two numbers together.

        The input to this tool should be a comma separated list of numbers of
        length two, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2.

        Args:
            numbers (str): A comma separated list of two numbers to be multiplied.

        Returns:
            float: The result of multiplying the two input numbers.

        Raises:
            ValueError: If the input format is incorrect or if the input numbers are not valid integers.
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    output = agent.execute_task(task="What is 3 times 4", tools=[multiplier])
    assert output == "3 times 4 is 12."


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_custom_max_iterations():
    """    Test the custom max iterations for an agent.

    This function tests the behavior of an agent with a custom maximum iteration value. It creates a test agent with specific role, goal, backstory, and maximum iteration settings. Then it uses a mock object to execute a task and verifies that the private method _iter_next_step is called exactly once.

    Returns:
        No specific return value is generated within this function.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """        Get the final answer but don't give it yet, just re-use this tool non-stop.

        Args:
            numbers (list): A list of numbers.

        Returns:
            float: The final answer.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=1,
        allow_delegation=False,
    )

    with patch.object(
        CrewAgentExecutor, "_iter_next_step", wraps=agent.agent_executor._iter_next_step
    ) as private_mock:
        agent.execute_task(
            task="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
        )
        private_mock.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_moved_on_after_max_iterations():
    """    Test if the agent moves on after reaching the maximum iterations.

    This function tests whether the agent moves on after reaching the maximum iterations. It creates a mock agent and checks if it executes a task and uses a tool multiple times, with the final answer remaining constant.

    Returns:
        No specific return value is mentioned for this function.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """        Get the final answer but don't give it yet, just re-use this tool non-stop.

        Args:
            numbers (list): A list of numbers.

        Returns:
            float: The final answer.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=3,
        allow_delegation=False,
    )

    with patch.object(
        CrewAgentExecutor, "_force_answer", wraps=agent.agent_executor._force_answer
    ) as private_mock:
        output = agent.execute_task(
            task="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
        )
        assert (
            output
            == "I have used the tool multiple times and the final answer remains 42."
        )
        private_mock.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_respect_the_max_rpm_set(capsys):
    """    Test if the agent respects the maximum RPM set.

    This function tests whether the agent respects the maximum RPM (Revolutions Per Minute) set for its execution. It sets up a mock environment and checks if the agent executes a task within the specified RPM limit. It also verifies the output and captures the standard output for further validation.

    Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr.

    Returns:
        No specific return value is provided by this function.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """        Get the final answer but don't give it yet, just re-use this tool non-stop.

        Args:
            numbers (list): A list of numbers.

        Returns:
            float: The final answer.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=5,
        max_rpm=1,
        verbose=True,
        allow_delegation=False,
    )

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        output = agent.execute_task(
            task="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
        )
        assert (
            output
            == "I've used the `get_final_answer` tool multiple times and it consistently returns the number 42."
        )
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_respect_the_max_rpm_set_over_crew_rpm(capsys):
    """    Test if the agent respects the maximum RPM set over the crew RPM.

    This function sets up a test scenario to check if the agent respects the maximum RPM set over the crew RPM. It creates an agent, a task, and a crew with specified parameters, and then uses a patch to simulate the behavior. After kicking off the crew, it captures the output and asserts certain conditions.

    Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr.

    Returns:
        No specific return value is provided by this function.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """        Get the final answer but don't give it yet, just re-use this tool non-stop.

        Args:
            numbers (list): A list of numbers.

        Returns:
            float: The final answer.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=4,
        max_rpm=10,
        verbose=True,
    )

    task = Task(
        description="Don't give a Final Answer, instead keep using the `get_final_answer` tool.",
        tools=[get_final_answer],
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], max_rpm=1, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." not in captured.out
        moveon.assert_not_called()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_without_max_rpm_respet_crew_rpm(capsys):
    """    Test the behavior of an agent when the maximum RPM is not respected by the crew.

    This function sets up a test scenario where two agents are created with specific roles, goals, backstories, and maximum RPM values. Tasks are assigned to these agents, and a crew is formed with these agents and tasks. The crew is then initiated, and the behavior of the agents is tested based on the maximum RPM constraint.

    Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr output.


    Raises:
        AssertionError: If the expected output does not match the actual captured output.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """        Get the final answer but don't give it yet, just re-use this tool non-stop.

        Args:
            numbers (list): A list of numbers.

        Returns:
            float: The final answer.
        """
        return 42

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_rpm=10,
        verbose=True,
    )

    agent2 = Agent(
        role="test role2",
        goal="test goal2",
        backstory="test backstory2",
        max_iter=2,
        verbose=True,
    )

    tasks = [
        Task(
            description="Just say hi.",
            agent=agent1,
        ),
        Task(
            description="Don't give a Final Answer, instead keep using the `get_final_answer` tool.",
            tools=[get_final_answer],
            agent=agent2,
        ),
    ]

    crew = Crew(agents=[agent1, agent2], tasks=tasks, max_rpm=1, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Action: get_final_answer" in captured.out
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_use_specific_tasks_output_as_context(capsys):
    """    Test the use of specific tasks' output as context.

    This function tests the use of specific tasks' output as context by creating two agents, assigning tasks to them, and then checking the output based on the assigned tasks.

    Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr.

    Returns:
        No specific return value is provided by this function.
    """

    pass

    agent1 = Agent(role="test role", goal="test goal", backstory="test backstory")

    agent2 = Agent(role="test role2", goal="test goal2", backstory="test backstory2")

    say_hi_task = Task(description="Just say hi.", agent=agent1)
    say_bye_task = Task(description="Just say bye.", agent=agent1)
    answer_task = Task(
        description="Answer accordingly to the context you got.",
        context=[say_hi_task],
        agent=agent2,
    )

    tasks = [say_hi_task, say_bye_task, answer_task]

    crew = Crew(agents=[agent1, agent2], tasks=tasks)
    result = crew.kickoff()
    assert "bye" not in result.lower()
    assert "hi" in result.lower() or "hello" in result.lower()
