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
    """    Test for conditional requirement in crew configuration.

    This function tests the conditional requirement in the crew configuration. It first checks if a ValueError is raised when creating a Crew object with a sequential process. Then it creates a sample configuration using JSON, creates a Crew object with the configuration, and asserts that the roles and descriptions of agents and tasks match the parsed configuration.


    Raises:
        ValueError: If the Crew object is created with an invalid process.
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
    """    Test for crew configuration with wrong keys.

    This function tests the behavior of the Crew class when provided with incorrect configuration keys. It checks for the appropriate raising of ValueError when the configuration contains wrong keys.

    Raises:
        ValueError: If the configuration contains wrong keys.
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
    """    Test the creation of a crew for handling tasks.

    This function creates a crew with specified agents, process, and tasks. It then kicks off the crew and checks if the result matches the expected output.

    Returns:
        The expected output of the crew kickoff, which includes a formatted list of ideas and their corresponding paragraphs.
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
    """    Test the hierarchical process by creating a task and a crew, then kicking off the crew to perform the task.

    This function creates a Task object with a specific description and then creates a Crew object with a list of agents, a process type, and a list of tasks. It then asserts that the crew's kickoff method returns a specific string.
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
    """    Test for creating a crew with delegating agents.

    This function sets up a test scenario where a crew is created with delegating agents and tasks. It then asserts the kickoff method of the crew and checks if it returns the expected result.

    Returns:
        The expected result of the crew's kickoff method.
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
    """    Test the verbose output of the Crew class.

    This function tests the verbose output of the Crew class by creating a crew with specific tasks and agents,
    setting the verbose flag to True, and then checking if the expected output strings are captured. It also tests
    the scenario when the verbose flag is set to False.

    Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr.

    Returns:
        No specific return value is provided by this function.
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
    """    Test the verbose levels output of the Crew class.

    This function sets up a Crew with a single task and a single agent, then tests the output of the Crew's kickoff method
    with different verbose levels. It captures the output using capsys and checks if the expected strings are present in
    the captured output.

    Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr.

    Returns:
        No specific return value for this function.
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
    """    Test cache hitting between agents.

    This function tests the cache handling between different agents in a crew. It creates a multiplier tool
    and assigns tasks to different agents. It then checks if the cache is empty initially, kicks off the crew,
    and verifies that the cache is updated with the result of the multiplier tool. It also mocks the cache read
    method to ensure proper cache handling.

    Returns:
        No specific return value is documented for this function.
    """

    from unittest.mock import patch

    from langchain.tools import tool

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
    """    Test the throttling of API calls.

    This function tests the throttling of API calls by simulating a scenario where a tool is repeatedly used without giving the final answer. It creates an agent, a task, and a crew, and then uses a mock object to simulate the passage of time and check if the maximum RPM is reached.

    Args:
        capsys: A built-in pytest fixture for capturing stdout and stderr.

    Returns:
        No specific return value is documented for this function.
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
