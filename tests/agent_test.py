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
    """Test the creation of an Agent object.

    This function creates an instance of the Agent class with specified
    role, goal, and backstory. It then asserts that the attributes of the
    created Agent object match the expected values. The test also checks
    that the initial list of tools is empty.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []


def test_agent_default_values():
    """Test the default values of an Agent instance.

    This function creates an Agent instance with specified role, goal, and
    backstory. It then checks that the LLM (Language Model) associated with
    the agent has the correct default values: - model_name should be
    "gpt-4". - temperature should be 0.7. - verbose should be False.
    Additionally, it verifies that allow_delegation is set to True.
    """

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0.7
    assert agent.llm.verbose == False
    assert agent.allow_delegation == True


def test_custom_llm():
    """Test the creation and properties of a custom LLM (Language Model) agent.

    This function creates an instance of an Agent with specific role, goal,
    backstory, and an OpenAI language model. It then verifies that the
    agent's language model is correctly set to an instance of OpenAI, with
    the specified model name and temperature.
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
    """Create and test agents with and without memory.

    This function demonstrates the creation of two agents: one with memory
    enabled and another without. It then executes a simple task to verify
    that the agent with memory retains its state, while the agent without
    memory does not.
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
    """Test the execution of a task by an agent.

    This function creates an instance of the `Agent` class with specific
    attributes and then calls its `execute_task` method to perform a simple
    arithmetic calculation. The test checks if the output matches the
    expected result.
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

    This function demonstrates how to create an agent with a custom tool and
    execute a task using that agent. The tool provided is a simple
    multiplier, which takes two numbers as input and returns their product.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiplies two numbers together.

        Args:
            numbers (str): A comma-separated string of two numeric values.

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
    """Test the usage of a logging tool within an agent.

    This function sets up an agent with a custom multiplication tool and
    asserts that the tool is correctly executed when given a task. It checks
    both the output and the tool usage information to ensure everything
    works as expected.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers together.

        Args:
            numbers (str): A comma separated list of two numeric values.

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
    """Test cache hitting functionality in an agent.

    This function tests the ability of an agent to handle task caching. It
    defines a simple tool, `multiplier`, which multiplies two numbers
    provided as a comma-separated string. The agent then executes several
    tasks, utilizing caching to store and retrieve results for repeated
    requests. The test checks that the cache is updated correctly after each
    task execution and that cached results are returned when available.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers together.

        Args:
            numbers (str): A comma-separated string of exactly two numeric values, representing the
                two numbers to be multiplied.

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
    """Test the execution of an agent using specific tools.
    This function sets up a test agent with a custom tool that multiplies
    two numbers. It then executes a task using this tool and asserts the
    expected output.
    """

    @tool
    def multiplier(numbers) -> float:
        """Multiplies two numbers together.

        Args:
            numbers (str): A comma-separated string of two numbers.

        Returns:
            float: The product of the two numbers.
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
    """Test the behavior of an agent when executing a task with a custom
    maximum iteration count.

    This function sets up an agent with a max_iter parameter set to 1 and
    executes a task that requires the use of a custom tool. The goal is to
    verify that the agent respects the max_iter constraint and only calls
    the specified tool once.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get a predetermined final answer without revealing it.

        This function is designed to simulate a process where repeated calls are
        made to a tool to obtain a final answer, but the actual calculation or
        logic is not exposed.

        Args:
            numbers (list): A list of numeric values, which are ignored in this context.

        Returns:
            float: The predetermined final answer, which is always 42.

        Note:
            This function's primary purpose is to demonstrate a non-revealing
            process for obtaining
            a predefined result. It should not be used for actual calculations or
            logic.
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
    """Test that the agent moves on after reaching the maximum number of
    iterations.

    This function tests the behavior of an agent when it reaches the
    specified maximum number of iterations. It uses a mock to ensure that
    the agent does not return an answer prematurely and instead continues
    using a tool until the maximum iteration limit is reached.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer by returning a constant value.

        This function is designed to simulate a scenario where the actual
        calculation or logic to determine the final answer is abstracted away.
        Instead, it always returns the same predefined value of 42.

        Args:
            numbers (list): A list of numeric values (though they are not used in this function).

        Returns:
            float: The constant value 42, representing the final answer.

        Note:
            This function is intended to be reused repeatedly without any changes,
            as it always returns the same result.
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
    """Test that the agent respects the max RPM set by not executing tasks
    faster than the allowed rate.

    This function sets up an agent with a maximum iterations and RPM
    (Revolutions Per Minute) limit. It then simulates the agent executing a
    task multiple times, using a tool that returns a constant value, to
    ensure that the agent does not exceed its RPM limit by waiting for the
    next minute before continuing.

    Args:
        capsys (pytest.fixture): Pytest fixture to capture console output.
    """

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-
        stop.

        This function is designed to generate a predetermined final answer (42)
        and is intended for use in scenarios where you need to repeatedly invoke
        a process without revealing the actual result. The purpose of this
        function is to test or demonstrate the behavior of a system under
        repeated calls.

        Args:
            numbers (list): A list of numeric values, which is not used in the computation but
                included as part of the function signature for consistency with other
                functions that might require such a parameter.

        Returns:
            float: The predetermined final answer, always 42.
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
    """Test agent respects max RPM set over crew RPM.

    This function sets up a scenario where an agent is tasked with a
    specific goal while respecting the maximum RPM limits imposed on both
    the agent and the crew. It uses mocking to control the behavior of the
    RPMController, ensuring that the agent does not proceed beyond its
    allowed RPM limit.

    Args:
        capsys (pytest_capture.CaptureFixture): Pytest fixture for capturing output.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-
        stop.

        Args:
            numbers (list): A list of numeric values. This parameter is not used in the function and
                exists for compatibility reasons.

        Returns:
            float: The value 42, representing a placeholder or dummy response.

        Note:
            The purpose of this function is to demonstrate the use of a tool that
            may be reused repeatedly without providing the actual final answer.
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
    """Test the behavior of an agent without a maximum RPM limit when
    interacting with a crew.

    This function simulates a scenario where two agents are part of a crew.
    One agent has a max_rpm limit, while the other does not. The test checks
    if the agent without the max_rpm limit respects the crew's max_rpm
    setting and handles it correctly by waiting for the next minute.

    Args:
        capsys (pytest_capture.CaptureFixture): A fixture to capture standard output.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get a predetermined final answer (42) regardless of the input.

        This function is designed to demonstrate a specific use case where a
        predefined value is returned without any consideration for the actual
        input parameters. It is intended to be reused repeatedly, hence its name
        "get_final_answer".

        Args:
            numbers (list): A list of numeric values (not used in this function).

        Returns:
            float: The fixed final answer value of 42.
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
    """Test that an agent uses specific tasks' outputs as context.

    This function creates two agents and assigns them specific tasks. It
    then runs a crew with these agents and tasks to verify that the output
    of one task is used as context for another task.

    Args:
        capsys (pytest.CaptureFixture): A fixture provided by pytest to capture the output of the function.
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
