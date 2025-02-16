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
    created Agent instance match the expected values. Additionally, it
    checks that the tools attribute is initialized as an empty list.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []


def test_agent_default_values():
    """Test the default values of the Agent class.

    This function creates an instance of the Agent class with predefined
    role, goal, and backstory values. It then asserts that the properties of
    the agent's llm (language model) are set to expected defaults. The
    assertions check the type of the llm, its model name, temperature,
    verbosity, and delegation allowance.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0.7
    assert agent.llm.verbose == False
    assert agent.allow_delegation == True


def test_custom_llm():
    """Test the initialization of a custom language model agent.

    This function creates an instance of the `Agent` class with specified
    role, goal, backstory, and a language model using OpenAI's GPT-4. It
    then asserts that the language model is correctly initialized with the
    expected parameters, ensuring that the agent is set up as intended.
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
    """Test the behavior of the Agent class with and without memory.

    This function creates two instances of the Agent class: one with memory
    disabled and another with memory enabled. It then executes a simple task
    using the no-memory agent and asserts the expected outcomes. The test
    checks that the no-memory agent does not retain any memory, while the
    memory-enabled agent does.
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

    This function creates an instance of the Agent class with predefined
    attributes such as role, goal, backstory, and delegation permissions. It
    then tests the agent's ability to execute a simple task by asking a
    mathematical question and asserting that the output is as expected.
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

    This function defines a tool that multiplies two numbers together and
    tests the agent's ability to execute a task using this tool. The tool
    accepts a comma-separated list of two numbers as input, splits the
    input, and returns the product of the two numbers. The agent is
    configured with a specific role, goal, and backstory, and is tasked with
    calculating the product of 3 and 4.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers provided as a comma-separated string.

        This function takes a string input consisting of two numbers separated
        by a comma, splits the string to extract the individual numbers, and
        returns their product. It is important that the input string contains
        exactly two numeric values; otherwise, the function may raise an error
        due to incorrect unpacking.

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
    """Test the usage of a logging tool with a multiplier function.

    This function defines a multiplier tool that takes a comma-separated
    list of two numbers as input and returns their product. It then creates
    an agent with specific role, goal, and backstory, and tests the
    execution of a task that utilizes the multiplier tool. The test checks
    if the output is correct and verifies that the tool usage is logged
    properly.
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
    an instance of the `Agent` class with the multiplier tool and executes
    several tasks to test the caching behavior. The function verifies that
    the cache is populated correctly with the results of the multiplication
    tasks and checks the output for specific queries. Additionally, it tests
    the behavior when the cache returns a default value.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers provided as a comma-separated string.

        This function takes a string of two comma-separated numbers, splits the
        string to extract the individual numbers, converts them to integers, and
        returns their product. It is important to ensure that the input string
        contains exactly two numbers, as the function does not handle cases with
        more or fewer numbers.

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

    This function sets up a test scenario for an agent that utilizes a
    multiplier tool to perform a multiplication task. The multiplier tool
    takes a comma-separated string of two numbers, splits the string,
    converts the numbers to integers, and returns their product. The agent
    is configured with a role, goal, and backstory, and is tasked with
    calculating the product of two numbers. The output is then asserted to
    ensure the agent's response is correct.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers provided as a comma-separated string.

        This function takes a string of two comma-separated numbers, splits the
        string to extract the individual numbers, converts them to integers, and
        returns their product. It is essential that the input string contains
        exactly two numbers; otherwise, the function may raise an error.

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

    This function sets up a test for an agent that is designed to execute a
    task using a specified tool. The agent is configured with a role, goal,
    backstory, and a maximum iteration limit. The test ensures that the
    agent correctly utilizes the tool to derive a final answer while
    adhering to the maximum iteration constraint. The function also verifies
    that the tool is called exactly once during the execution of the task.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer based on the provided numbers.

        This function is designed to return a predetermined value, which is 42,
        regardless of the input. The purpose of this function is to serve as a
        placeholder or a tool that can be reused in various contexts without
        providing an actual computation based on the input numbers.

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
    """Test the behavior of an agent when it reaches the maximum number of
    iterations.

    This function tests whether the agent correctly handles a scenario where
    it is restricted to a maximum number of iterations. It utilizes a tool
    that provides a final answer but is designed to be reused multiple times
    without revealing the answer immediately. The test verifies that the
    agent can execute a task that involves using this tool and checks if the
    expected output is produced after the maximum iterations are reached.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer based on the provided numbers.

        This function is designed to return a predetermined value (42)
        regardless of the input. It serves as a placeholder or a tool for
        further development, allowing for continuous reuse without providing an
        actual answer based on the input.

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
    """Test that the agent respects the maximum RPM set during task execution.

    This function tests the behavior of an agent when it is tasked with
    using a tool that provides a final answer. The agent is configured with
    a maximum RPM (requests per minute) limit, and the test verifies that
    the agent adheres to this limit while executing the task. The agent
    should utilize the provided tool multiple times without exceeding the
    specified RPM, and the output should confirm the consistent return value
    from the tool.

    Args:
        capsys (pytest.CaptureFixture): A fixture that captures standard output and
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer based on the provided numbers.

        This function is a placeholder that returns a constant value of 42,
        regardless of the input. It is intended to demonstrate the structure of
        a function that would eventually compute a final answer based on the
        provided list of numbers. The actual logic for processing the numbers is
        not implemented in this version.

        Args:
            numbers (list): A list of numeric values to be processed.

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

    This function tests the behavior of an agent within a crew to ensure
    that the agent does not exceed the maximum RPM (Rounds Per Minute) limit
    set for the crew. It sets up a mock environment with a tool that
    provides a final answer but is designed not to give it immediately. The
    agent is configured with specific parameters, including a maximum RPM
    limit. The test then initiates the crew's tasks and checks that the
    output does not indicate that the maximum RPM has been reached, ensuring
    that the agent behaves correctly within the defined constraints.

    Args:
        capsys (pytest.capsys): A pytest fixture that captures standard output
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer based on the provided numbers.

        This function is a placeholder that returns a constant value of 42,
        regardless of the input. It serves as a demonstration of a function that
        is intended to provide a final answer but does not currently utilize the
        input numbers in any meaningful way.

        Args:
            numbers (list): A list of numeric values that are intended to

        Returns:
            float: The constant value 42, representing the final answer.
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
    """Test the behavior of an agent without a maximum RPM in respect to crew
    RPM.

    This function sets up a test scenario using two agents and a crew, where
    one agent has a defined maximum RPM and the other does not. The test
    verifies that the crew correctly handles the situation when the maximum
    RPM is reached, ensuring that the appropriate actions are taken and
    output messages are generated. The `get_final_answer` tool is used by
    one of the agents, and the test checks if the expected output is
    produced when the crew is kicked off.

    Args:
        capsys (pytest.CaptureFixture): A fixture provided by pytest to capture output during the test.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer based on the provided numbers.

        This function is designed to return a constant value of 42, regardless
        of the input. It serves as a placeholder or a tool that can be reused in
        various contexts without providing an actual computation based on the
        input numbers.

        Args:
            numbers (list): A list of numeric values that are not used

        Returns:
            float: The constant value 42.
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

    This function tests the interaction between two agents, where one agent
    performs tasks that involve greeting and farewelling, while the other
    agent responds based on the context provided by the first agent's tasks.
    It asserts that the response does not contain a farewell and includes a
    greeting.

    Args:
        capsys: A pytest fixture that captures standard output and error.
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
