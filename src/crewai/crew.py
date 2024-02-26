import json
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    Json,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.process import Process
from crewai.task import Task
from crewai.tools.agent_tools import AgentTools
from crewai.utilities import I18N, Logger, RPMController


class Crew(BaseModel):
    """
    Represents a group of agents, defining how they should collaborate and the tasks they should perform.

    Attributes:
        tasks: List of tasks assigned to the crew.
        agents: List of agents part of this crew.
        process: The process flow that the crew will follow (e.g., sequential).
        verbose: Indicates the verbosity level for logging during execution.
        config: Configuration settings for the crew.
        _cache_handler: Handles caching for the crew's operations.
        max_rpm: Maximum number of requests per minute for the crew execution to be respected.
        id: A unique identifier for the crew instance.
    """

    __hash__ = object.__hash__  # type: ignore
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _cache_handler: InstanceOf[CacheHandler] = PrivateAttr(default=CacheHandler())
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[Agent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: Union[int, bool] = Field(default=0)
    config: Optional[Union[Json, Dict[str, Any]]] = Field(default=None)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the crew execution to be respected.",
    )
    language: str = Field(
        default="en",
        description="Language used for the crew, defaults to English.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """        Prevent manual setting of the 'id' field by users.

        This function prevents users from manually setting the 'id' field. If a value is provided for the 'id' field,
        it raises a PydanticCustomError with the message "The 'id' field cannot be set by the user."

        Args:
            cls: The class instance.
            v (Optional[UUID4]): The value for the 'id' field.


        Raises:
            PydanticCustomError: If a value is provided for the 'id' field.
        """
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "The 'id' field cannot be set by the user.", {}
            )

    @field_validator("config", mode="before")
    @classmethod
    def check_config_type(
        cls, v: Union[Json, Dict[str, Any]]
    ) -> Union[Json, Dict[str, Any]]:
        """        Validates that the config is a valid type.

        Args:
            cls: The class instance.
            v: The config to be validated.

        Returns:
            The config if it is valid.
        """

        # TODO: Improve typing
        return json.loads(v) if isinstance(v, Json) else v  # type: ignore

    @model_validator(mode="after")
    def set_private_attrs(self) -> "Crew":
        """        Set private attributes.

        This method initializes and sets the private attributes _cache_handler, _logger, and _rpm_controller.
        It creates a new CacheHandler instance and assigns it to _cache_handler attribute.
        It creates a new Logger instance with the given verbosity level and assigns it to _logger attribute.
        It creates a new RPMController instance with the given max_rpm and logger attributes and assigns it to _rpm_controller attribute.

        Returns:
            Crew: The current instance of the Crew class.
        """
        self._cache_handler = CacheHandler()
        self._logger = Logger(self.verbose)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        return self

    @model_validator(mode="after")
    def check_config(self):
        """        Validates that the crew is properly configured with agents and tasks.

        This method checks if the crew is properly configured with agents and tasks. If the configuration is missing or incomplete,
        it raises a PydanticCustomError with a specific message. It then sets up the crew from the configuration and updates the cache
        handler and RPM controller for each agent. Finally, it returns the current instance of the crew.

        Returns:
            Crew: The current instance of the crew.

        Raises:
            PydanticCustomError: If the configuration is missing or incomplete.
        """
        if not self.config and not self.tasks and not self.agents:
            raise PydanticCustomError(
                "missing_keys",
                "Either 'agents' and 'tasks' need to be set or 'config'.",
                {},
            )

        if self.config:
            self._setup_from_config()

        if self.agents:
            for agent in self.agents:
                agent.set_cache_handler(self._cache_handler)
                agent.set_rpm_controller(self._rpm_controller)
        return self

    def _setup_from_config(self):
        """        Initializes agents and tasks from the provided config.

        Raises:
            PydanticCustomError: If 'agents' or 'tasks' are missing in the config.
        """

        assert self.config is not None, "Config should not be None."

        """Initializes agents and tasks from the provided config."""
        if not self.config.get("agents") or not self.config.get("tasks"):
            raise PydanticCustomError(
                "missing_keys_in_config", "Config should have 'agents' and 'tasks'.", {}
            )

        self.agents = [Agent(**agent) for agent in self.config["agents"]]
        self.tasks = [self._create_task(task) for task in self.config["tasks"]]

    def _create_task(self, task_config: Dict[str, Any]) -> Task:
        """        Creates a task instance from its configuration.

        Args:
            task_config (Dict[str, Any]): The configuration of the task.

        Returns:
            Task: A task instance.
        """
        task_agent = next(
            agt for agt in self.agents if agt.role == task_config["agent"]
        )
        del task_config["agent"]
        return Task(**task_config, agent=task_agent)

    def kickoff(self) -> str:
        """        Starts the crew to work on its assigned tasks.

        This method initializes the crew members with the specified language and then starts the process based on the assigned process type.

        Returns:
            str: The result of the process execution.

        Raises:
            NotImplementedError: If the specified process type is not implemented.
        """
        for agent in self.agents:
            agent.i18n = I18N(language=self.language)

        if self.process == Process.sequential:
            return self._run_sequential_process()
        if self.process == Process.hierarchical:
            return self._run_hierarchical_process()

        raise NotImplementedError(
            f"The process '{self.process}' is not implemented yet."
        )

    def _run_sequential_process(self) -> str:
        """        Executes tasks sequentially and returns the final output.

        This method iterates through the tasks in a sequential manner, performing each task and updating the task_output
        variable with the result. If the task's agent allows delegation, it adds tools from the agents to the task. It
        logs the working agent and the task being started. After executing each task, it logs the task output and the
        working agent's role. If max_rpm is set, it stops the RPM counter.

        Returns:
            str: The final output after executing all tasks.
        """
        task_output = ""
        for task in self.tasks:
            if task.agent is not None and task.agent.allow_delegation:
                task.tools += AgentTools(agents=self.agents).tools()

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"Working Agent: {role}")
            self._logger.log("info", f"Starting Task: {task.description}")

            task_output = task.execute(context=task_output)

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"[{role}] Task output: {task_output}\n\n")

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return task_output

    def _run_hierarchical_process(self) -> str:
        """        Creates and assigns a manager agent to make sure the crew completes the tasks.

        This method creates a manager agent using the provided language and assigns it to oversee the completion of the tasks. It iterates through each task in the list of tasks, executing them using the manager agent. If a maximum RPM (Revolutions Per Minute) is specified, it stops the RPM counter before returning the final task output.

        Returns:
            str: The final task output after all tasks have been executed.
        """

        i18n = I18N(language=self.language)
        manager = Agent(
            role=i18n.retrieve("hierarchical_manager_agent", "role"),
            goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
            backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
            tools=AgentTools(agents=self.agents).tools(),
            verbose=True,
        )

        task_output = ""
        for task in self.tasks:
            self._logger.log("info", f"Starting Task: {task.description}")

            task_output = task.execute(agent=manager, context=task_output)

            self._logger.log(
                "debug", f"[{manager.role}] Task output: {task_output}\n\n"
            )

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return task_output
