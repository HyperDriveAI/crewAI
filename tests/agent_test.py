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
    """Test the creation of an Agent instance.

    This function creates an Agent object with predefined attributes such as
    role, goal, and backstory. It then asserts that the attributes of the
    created Agent match the expected values. Additionally, it verifies that
    the tools attribute is initialized as an empty list.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []


def test_agent_default_values():
    """Test the default values of the Agent class.

    This function creates an instance of the Agent class with predefined
    role, goal, and backstory values. It then asserts that the default
    properties of the agent's LLM (Language Learning Model) are set
    correctly, including the model name, temperature, verbosity, and
    delegation allowance.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0.7
    assert agent.llm.verbose == False
    assert agent.allow_delegation == True


def test_custom_llm():
    """Test the initialization of a custom LLM agent.

    This function creates an instance of the Agent class with specified
    role, goal, backstory, and a language model (LLM) using OpenAI's GPT-4.
    It then asserts that the LLM is correctly initialized with the expected
    model name and temperature settings.
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
    """Test the behavior of an agent with and without memory.

    This function creates two instances of the Agent class: one without
    memory and one with memory. It then executes a simple task using the
    agent without memory and asserts that the result is correct.
    Additionally, it verifies that the memory attribute of the no-memory
    agent is None, while the memory attribute of the memory agent is not
    None.
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
    """Test the execution of an agent's task.

    This function creates an instance of the Agent class with specified
    role, goal, backstory, and delegation settings. It then tests the
    agent's ability to execute a simple arithmetic task by asserting that
    the output of the task matches the expected result.
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
    """Test the execution of an agent with tools.

    This function defines a tool called `multiplier`, which is designed to
    multiply two numbers provided as a comma-separated string. The agent is
    then created with specific attributes and tasked to execute a
    multiplication operation. The output is asserted to ensure that the
    agent correctly performs the multiplication.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers together.

        This function takes a comma-separated string of two numeric values,
        splits the string to extract the individual numbers, and then multiplies
        them together. It is important that the input string contains exactly
        two numbers; otherwise, the function may raise an error.

        Args:
            numbers (str): A comma-separated string containing two numeric values.

        Returns:
            float: The product of the two input numbers.
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
    """Test the usage of the logging tool with a multiplier function.

    This function defines a tool called `multiplier` that takes a comma-
    separated list of two numbers as input and returns their product. The
    test creates an agent with specific role, goal, and backstory, and then
    executes a task to multiply two numbers. The assertions check that the
    tool was used correctly and that the output matches the expected result.
    The `multiplier` tool is decorated with `@tool`, indicating that it is a
    function designed to be used as a tool within the agent's context.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers provided as a comma-separated string.

        This function takes a string input containing two numbers separated by a
        comma, splits the string to extract the individual numbers, converts
        them to integers, and returns their product. It is important that the
        input string contains exactly two numeric values; otherwise, the
        function may raise an error.

        Args:
            numbers (str): A comma-separated string containing two numeric values.

        Returns:
            float: The product of the two input numbers.
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
    """Test the functionality of the cache hitting mechanism in the agent.

    This function defines a tool called `multiplier`, which takes a comma-
    separated list of two numbers and returns their product. It then creates
    an instance of the `Agent` class with specific parameters and executes
    several tasks to test the caching behavior of the agent when performing
    multiplication operations. The results are asserted against expected
    values to ensure that the cache is functioning correctly.  The test
    checks if the cache correctly stores and retrieves results for
    previously computed multiplications, and it also verifies that the cache
    returns the expected value when a cached result is requested.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers provided as a comma-separated string.

        This function takes a string input containing two numbers separated by a
        comma, splits the string to extract the individual numbers, converts
        them to integers, and returns their product. It is important that the
        input string contains exactly two numbers; otherwise, the function may
        raise an error.

        Args:
            numbers (str): A comma-separated string containing exactly two numeric values.

        Returns:
            float: The product of the two input numbers.
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
    """Test the execution of an agent with specific tools.

    This function defines a tool called `multiplier`, which is designed to
    multiply two numbers provided as a comma-separated string. The agent is
    then created with a specified role, goal, and backstory, and tasked to
    execute a multiplication operation. The output of the agent's task
    execution is asserted to ensure it matches the expected result.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers together.

        This function takes a comma-separated string of two numeric values,
        splits the string to extract the individual numbers, converts them to
        integers, and returns their product. It is important that the input
        string contains exactly two numbers separated by a comma. For example,
        an input of `1,2` will return the product of 1 and 2.

        Args:
            numbers (str): A comma-separated string containing two numeric values.

        Returns:
            float: The product of the two input numbers.
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
    """Test the agent's behavior with a custom maximum iteration limit.

    This function defines a test case for an agent that is configured to
    execute a task with a maximum of one iteration. It utilizes a tool that
    provides a final answer but is designed to be called repeatedly without
    returning the answer immediately. The test verifies that the agent
    correctly calls the tool only once during the execution of the task.
    The agent is initialized with specific role, goal, backstory, and
    iteration settings. The test uses mocking to ensure that the agent's
    execution flow behaves as expected.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-
        stop.

        This function is designed to return a predetermined final answer, which
        is currently hardcoded as 42. The purpose of this function is to serve
        as a placeholder or a tool that can be reused in various contexts
        without revealing the actual answer. It accepts a parameter but does not
        utilize it in the current implementation.

        Args:
            numbers (list): A list of numeric values (not used in the current implementation).

        Returns:
            float: The final answer, which is always 42.
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
    """Test the behavior of the agent when it reaches the maximum number of
    iterations.

    This function tests whether the agent correctly utilizes a tool to
    derive an answer without prematurely revealing it. It sets up an agent
    with specific parameters and executes a task that involves using the
    tool multiple times. The test asserts that the output is as expected and
    verifies that the tool was called only once during the execution.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-
        stop.

        This function is designed to return a predefined answer, which is 42,
        regardless of the input provided. It serves as a placeholder or a
        constant value that can be reused in various contexts where a final
        answer is required without performing any calculations based on the
        input numbers.

        Args:
            numbers (list): A list of numeric values (not used in the current implementation).

        Returns:
            float: The final answer, which is always 42.
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
    """Test the agent's adherence to the maximum RPM limit set during
    execution.

    This function tests whether the agent respects the maximum RPM (requests
    per minute) limit when executing a task. It defines a tool that returns
    a constant value and creates an agent with specific parameters. The
    agent is then tasked with using the tool multiple times, and the output
    is verified against the expected result. Additionally, it checks that
    the appropriate message is captured when the maximum RPM is reached.

    Args:
        capsys (pytest.CaptureFixture): A fixture that captures standard output and error.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer based on the provided numbers.

        This function is designed to return a constant value of 42, regardless
        of the input. It serves as a placeholder or a tool that can be reused in
        various contexts without providing any specific calculations based on
        the input numbers.

        Args:
            numbers (list): A list of numeric values that are not used

        Returns:
            float: The constant value 42.
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
    """Test that the agent respects the maximum RPM set over the crew RPM.

    This function tests the behavior of an agent when it is configured with
    a maximum RPM (revolutions per minute) limit. It sets up a mock tool
    that returns a final answer but is designed not to provide it
    immediately. The test ensures that when the crew's RPM is set to a lower
    value than the agent's maximum RPM, the agent does not exceed this limit
    during execution. The test captures the output to verify that the
    expected behavior is followed and that the agent waits appropriately
    without exceeding the RPM constraints.

    Args:
        capsys (pytest fixture): A pytest fixture that allows capturing
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer without revealing it immediately.

        This function is designed to provide a consistent final answer for a
        given set of numbers. While the input is accepted, the function does not
        utilize it in any calculations and simply returns a predetermined value.
        This can be useful in scenarios where a constant result is needed
        regardless of the input.

        Args:
            numbers (list): A list of numeric values that are not used

        Returns:
            float: The final answer, which is always 42.
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
    """Test the behavior of agents when the maximum RPM is not respected.

    This test simulates a scenario where two agents are created with
    specific roles, goals, and backstories. The first agent has a defined
    maximum RPM, while the second agent utilizes a tool that is designed to
    provide a final answer without actually delivering it immediately. The
    test checks if the crew can handle the tasks assigned to the agents and
    verifies that the output correctly reflects the actions taken by the
    agents, particularly focusing on the RPM control mechanism.

    Args:
        capsys (pytest.capsys): A pytest fixture that captures output to stdout and stderr.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-
        stop.

        This function is designed to return a predefined final answer, which is
        currently set to 42. The purpose of this function is to serve as a
        placeholder or a tool that can be reused in various contexts without
        providing the actual answer immediately. It accepts a list of numbers as
        input, but the input does not affect the output, which remains constant.

        Args:
            numbers (list): A list of numeric values that are not utilized in the

        Returns:
            float: The final answer, which is always 42.
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
    """Test the output of agents using specific tasks as context.

    This test function creates two agents with distinct roles and goals,
    assigns them specific tasks, and then initiates a crew to execute those
    tasks. The purpose of this test is to verify that the output generated
    by the agents adheres to the expected behavior based on the tasks
    assigned. Specifically, it checks that the output does not include the
    word "bye" while ensuring that either "hi" or "hello" is present in the
    output.

    Args:
        capsys (pytest.capsys): A pytest fixture that captures
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
