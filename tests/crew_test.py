"""Test Agent creation and execution basic functionality."""

import json

import pytest

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task
from crewai.utilities import Logger, RPMController

ceo = Agent(
    role="CEO",
    goal="Make sure the writers in your company produce amazing content.",
    backstory="You're an long time CEO of a content creation agency with a Senior Writer on the team. You're now working on a new project and want to make sure the content produced is amazing.",
    allow_delegation=True,
)

researcher = Agent(
    role="Researcher",
    goal="Make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Write the best content about AI and AI agents.",
    backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
    allow_delegation=False,
)


def test_crew_config_conditional_requirement():
    """Test that a ValueError is raised when creating a Crew instance with an
    empty config.

    This function uses PyTest to assert that attempting to create a Crew
    instance without a valid configuration raises a ValueError. The function
    also checks if the Crew instance is created correctly when provided with
    a valid JSON configuration.
    """

    with pytest.raises(ValueError):
        Crew(process=Process.sequential)

    config = json.dumps(
        {
            "agents": [
                {
                    "role": "Senior Researcher",
                    "goal": "Make the best research and analysis on content about AI and AI agents",
                    "backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
                },
                {
                    "role": "Senior Writer",
                    "goal": "Write the best content about AI and AI agents.",
                    "backstory": "You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
                },
            ],
            "tasks": [
                {
                    "description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
                    "agent": "Senior Researcher",
                },
                {
                    "description": "Write a 1 amazing paragraph highlight for each idead that showcases how good an article about this topic could be, check references if necessary or search for more content but make sure it's unique, interesting and well written. Return the list of ideas with their paragraph and your notes.",
                    "agent": "Senior Writer",
                },
            ],
        }
    )
    parsed_config = json.loads(config)

    try:
        crew = Crew(process=Process.sequential, config=config)
    except ValueError:
        pytest.fail("Unexpected ValidationError raised")

    assert [agent.role for agent in crew.agents] == [
        agent["role"] for agent in parsed_config["agents"]
    ]
    assert [task.description for task in crew.tasks] == [
        task["description"] for task in parsed_config["tasks"]
    ]


def test_crew_config_with_wrong_keys():
    """Test the Crew class initialization with configurations containing wrong
    keys.

    The function creates three different configuration JSON strings: -
    `no_tasks_config`: A valid configuration without tasks. -
    `no_agents_config`: A valid configuration without agents. - An invalid
    configuration with a single key-value pair.  It then tests the Crew
    class initialization using these configurations and asserts that it
    raises a ValueError in each case.
    """

    no_tasks_config = json.dumps(
        {
            "agents": [
                {
                    "role": "Senior Researcher",
                    "goal": "Make the best research and analysis on content about AI and AI agents",
                    "backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
                }
            ]
        }
    )

    no_agents_config = json.dumps(
        {
            "tasks": [
                {
                    "description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
                    "agent": "Senior Researcher",
                }
            ]
        }
    )
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config='{"wrong_key": "wrong_value"}')
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config=no_tasks_config)
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config=no_agents_config)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_creation():
    """Create and kick off a crew with specific tasks for generating ideas and
    highlighting their value.

    This function initializes a Crew object with predefined tasks, each
    focusing on creating interesting article ideas and providing compelling
    highlights. The crew's process is set to sequential execution, ensuring
    that one task completes before moving on to the next. The kickoff method
    of the crew is then called, which should return a string detailing the
    generated content.

    Returns:
        str: A formatted string containing five unique and intriguing ideas for
            articles, each accompanied by a compelling highlight.
    """

    tasks = [
        Task(
            description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
            agent=researcher,
        ),
        Task(
            description="Write a 1 amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
            agent=writer,
        ),
    ]

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    assert (
        crew.kickoff()
        == """1. **The Evolution of AI: From Old Concepts to New Frontiers** - Journey with us as we traverse the fascinating timeline of artificial intelligence - from its philosophical and mathematical infancy to the sophisticated, problem-solving tool it has become today. This riveting account will not only educate but also inspire, as we delve deep into the milestones that brought us here and shine a beacon on the potential that lies ahead.

2. **AI Agents in Healthcare: The Future of Medicine** - Imagine a world where illnesses are diagnosed before symptoms appear, where patient outcomes are not mere guesses but accurate predictions. This is the world AI is crafting in healthcare - a revolution that's saving lives and changing the face of medicine as we know it. This article will spotlight this transformative journey, underlining the profound impact AI is having on our health and well-being.

3. **AI and Ethics: Navigating the Moral Landscape of Artificial Intelligence** - As AI becomes an integral part of our lives, it brings along a plethora of ethical dilemmas. This thought-provoking piece will navigate the complex moral landscape of AI, addressing critical concerns like privacy, job displacement, and decision-making biases. It serves as a much-needed discussion platform for the societal implications of AI, urging us to look beyond the technology and into the mirror.

4. **Demystifying AI Algorithms: A Deep Dive into Machine Learning** - Ever wondered what goes on behind the scenes of AI? This enlightening article will break down the complex world of machine learning algorithms into digestible insights, unraveling the mystery of AI's 'black box'. It's a rare opportunity for the non-technical audience to appreciate the inner workings of AI, fostering a deeper understanding of this revolutionary technology.

5. **AI Startups: The Game Changers of the Tech Industry** - In the world of tech, AI startups are the bold pioneers charting new territories. This article will spotlight these game changers, showcasing how their innovative products and services are driving the AI revolution. It's a unique opportunity to catch a glimpse of the entrepreneurial side of AI, offering inspiration for the tech enthusiasts and dreamers alike."""
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_hierarchical_process():
    """Test the hierarchical processing workflow for a given task.

    This function sets up a hierarchical process involving a crew with
    multiple agents (researcher, writer) and executes the kickoff method. It
    asserts that the output matches the expected formatted string containing
    5 interesting ideas with their highlight paragraphs.
    """

    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        tasks=[task],
    )

    assert (
        crew.kickoff()
        == """Here are the 5 interesting ideas with a highlight paragraph for each:

1. "The Future of AI in Healthcare: Predicting Diseases Before They Happen"
   - "Imagine a future where AI empowers us to detect diseases before they arise, transforming healthcare from reactive to proactive. Machine learning algorithms, trained on vast amounts of patient data, could potentially predict heart diseases, strokes, or cancers before they manifest, allowing for early interventions and significantly improving patient outcomes. This article will delve into the rapid advancements in AI within the healthcare sector and how these technologies are ushering us into a new era of predictive medicine."

2. "How AI is Changing the Way We Cook: An Insight into Smart Kitchens"
   - "From the humble home kitchen to grand culinary stages, AI is revolutionizing the way we cook. Smart appliances, equipped with advanced sensors and predictive algorithms, are turning kitchens into creative playgrounds, offering personalized recipes, precise cooking instructions, and even automated meal preparation. This article explores the fascinating intersection of AI and gastronomy, revealing how technology is transforming our culinary experiences."

3. "Redefining Fitness with AI: Personalized Workout Plans and Nutritional Advice"
   - "Fitness reimagined – that's the promise of AI in the wellness industry. Picture a personal trainer who knows your strengths, weaknesses, and nutritional needs intimately. An AI-powered fitness app can provide this personalized experience, adapting your workout plans and dietary recommendations in real-time based on your progress and feedback. Join us as we unpack how AI is revolutionizing the fitness landscape, offering personalized, data-driven approaches to health and well-being."

4. "AI and the Art World: How Technology is Shaping Creativity"
   - "Art and AI may seem like unlikely partners, but their synergy is sparking a creative revolution. AI algorithms are now creating mesmerizing artworks, challenging our perceptions of creativity and originality. From AI-assisted painting to generative music composition, this article will take you on a journey through the fascinating world of AI in art, exploring how technology is reshaping the boundaries of human creativity."

5. "AI in Space Exploration: The Next Frontier"
   - "The vast expanse of space, once the sole domain of astronauts and rovers, is the next frontier for AI. AI technology is playing an increasingly vital role in space exploration, from predicting space weather to assisting in interstellar navigation. This article will delve into the exciting intersection of AI and space exploration, exploring how these advanced technologies are helping us uncover the mysteries of the cosmos.\""""
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_with_delegating_agents():
    """Test a crew with delegating agents by creating tasks and kicking off the
    process.

    This function sets up a crew with specific agents and tasks, then kicks
    off the process to generate an article draft. It asserts that the output
    matches the expected result.
    """

    tasks = [
        Task(
            description="Produce and amazing 1 paragraph draft of an article about AI Agents.",
            agent=ceo,
        )
    ]

    crew = Crew(
        agents=[ceo, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    assert (
        crew.kickoff()
        == '"AI agents, the digital masterminds at the heart of the 21st-century revolution, are shaping a new era of intelligence and innovation. They are autonomous entities, capable of observing their environment, making decisions, and acting on them, all in pursuit of a specific goal. From streamlining operations in logistics to personalizing customer experiences in retail, AI agents are transforming how businesses operate. But their potential extends far beyond the corporate world. They are the sentinels protecting our digital frontiers, the virtual assistants making our lives easier, and the unseen hands guiding autonomous vehicles. As this technology evolves, AI agents will play an increasingly central role in our world, ushering in an era of unprecedented efficiency, personalization, and productivity. But with great power comes great responsibility, and understanding and harnessing this potential responsibly will be one of our greatest challenges and opportunities in the coming years."'
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_verbose_output(capsys):
    """Test the verbose output functionality of the Crew class.

    This function tests that the Crew class correctly outputs messages based
    on the verbosity level. It creates a crew with two tasks and asserts
    that the expected debug and info messages are present when verbose is
    True. Then, it resets the logger to non-verbose mode and asserts that no
    output is produced.

    Args:
        capsys (pytest fixtures): Captures stdout and stderr during test execution.
    """

    tasks = [
        Task(description="Research AI advancements.", agent=researcher),
        Task(description="Write about AI in healthcare.", agent=writer),
    ]

    crew = Crew(
        agents=[researcher, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = [
        "[DEBUG]: Working Agent: Researcher",
        "[INFO]: Starting Task: Research AI advancements.",
        "[DEBUG]: [Researcher] Task output:",
        "[DEBUG]: Working Agent: Senior Writer",
        "[INFO]: Starting Task: Write about AI in healthcare.",
        "[DEBUG]: [Senior Writer] Task output:",
    ]

    for expected_string in expected_strings:
        assert expected_string in captured.out

    # Now test with verbose set to False
    crew._logger = Logger(verbose_level=False)
    crew.kickoff()
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_verbose_levels_output(capsys):
    """Test the verbose levels of crew output.

    This function tests the output of a crew instance with different verbose
    levels. It asserts that specific strings are present in the captured
    output when the crew is kicked off with various verbosity settings.

    Args:
        capsys (pytest.CaptureFixture): The fixture to capture standard output and standard error during test
            execution.
    """

    tasks = [Task(description="Write about AI advancements.", agent=researcher)]

    crew = Crew(agents=[researcher], tasks=tasks, process=Process.sequential, verbose=1)

    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = ["Working Agent: Researcher", "[Researcher] Task output:"]

    for expected_string in expected_strings:
        assert expected_string in captured.out

    # Now test with verbose set to 2
    crew._logger = Logger(verbose_level=2)
    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = [
        "Working Agent: Researcher",
        "Starting Task: Write about AI advancements.",
        "[Researcher] Task output:",
    ]

    for expected_string in expected_strings:
        assert expected_string in captured.out


@pytest.mark.vcr(filter_headers=["authorization"])
def test_cache_hitting_between_agents():
    """Test the behavior of cache hitting between agents in a Crew object.

    This function sets up a scenario with multiple tasks and agents, where
    one task involves using a tool to multiply two numbers. The function
    then asserts that the cache is correctly handled when the same task is
    executed again, demonstrating how cached results are reused.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def multiplier(numbers) -> float:
        """Multiply two numbers together.

        Args:
            numbers (str): A comma-separated string of two numeric values.

        Returns:
            float: The product of the two input numbers.
        """
        a, b = numbers.split(",")
        return int(a) * int(b)

    tasks = [
        Task(
            description="What is 2 tims 6? Return only the number.",
            tools=[multiplier],
            agent=ceo,
        ),
        Task(
            description="What is 2 times 6? Return only the number.",
            tools=[multiplier],
            agent=researcher,
        ),
    ]

    crew = Crew(
        agents=[ceo, researcher],
        tasks=tasks,
    )

    assert crew._cache_handler._cache == {}
    output = crew.kickoff()
    assert crew._cache_handler._cache == {"multiplier-2,6": "12"}
    assert output == "12"

    with patch.object(CacheHandler, "read") as read:
        read.return_value = "12"
        crew.kickoff()
        read.assert_called_with("multiplier", "2,6")


@pytest.mark.vcr(filter_headers=["authorization"])
def test_api_calls_throttling(capsys):
    """Test the throttling behavior of API calls using a mock RPMController.

    This function simulates a scenario where an agent repeatedly calls a
    tool (`get_final_answer`) until the maximum RPM (requests per minute) is
    reached. It uses a mocked RPMController to control the throttling
    behavior and captures the output to verify that the throttling message
    is printed when the limit is exceeded.

    Args:
        capsys: Pytest fixture to capture standard output.
    """

    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-
        stop.

        This function returns a predetermined value of 42. It's designed to be
        repeatedly used, making it useful for testing or as a placeholder in a
        larger system.

        Args:
            numbers (list): A list of numeric values that is ignored by the function.

        Returns:
            float: The constant value 42.
        """
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=5,
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description="Don't give a Final Answer, instead keep using the `get_final_answer` tool.",
        tools=[get_final_answer],
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], max_rpm=2, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()
