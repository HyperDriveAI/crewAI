import uuid
from typing import Any, List, Optional

from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.tasks.task_output import TaskOutput
from crewai.utilities import I18N


class Task(BaseModel):
    """Class that represent a task to be executed."""

    __hash__ = object.__hash__  # type: ignore
    i18n: I18N = I18N()
    description: str = Field(description="Description of the actual task.")
    callback: Optional[Any] = Field(
        description="Callback to be executed after the task is completed.", default=None
    )
    agent: Optional[Agent] = Field(
        description="Agent responsible for execution the task.", default=None
    )
    expected_output: Optional[str] = Field(
        description="Clear definition of expected output for the task.",
        default=None,
    )
    context: Optional[List["Task"]] = Field(
        description="Other tasks that will have their output used as context for this task.",
        default=None,
    )
    output: Optional[TaskOutput] = Field(
        description="Task output, it's final result after being executed", default=None
    )
    tools: List[Any] = Field(
        default_factory=list,
        description="Tools the agent is limited to use for this task.",
    )
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """        Deny user from setting the ID.

        This function checks if the user is trying to set the ID and raises a PydanticCustomError
        with the message "This field is not to be set by the user." if the value is not None.

        Args:
            cls: The class instance.
            v: The value of the ID.


        Raises:
            PydanticCustomError: If the value is not None.
        """

        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def check_tools(self):
        """        Check if the tools are set.

        This method checks if the tools are set. If the tools are not set and the agent and its tools are available,
        it extends the tools list with the agent's tools.

        Returns:
            self: The instance of the class after checking the tools.
        """
        if not self.tools and self.agent and self.agent.tools:
            self.tools.extend(self.agent.tools)
        return self

    def execute(self, agent: Agent | None = None, context: Optional[str] = None) -> str:
        """        Execute the task.

        This method executes the task using the provided agent and context, and returns the output of the task.

        Args:
            agent (Agent?): The agent to be used for executing the task. If not provided, the default agent assigned to the task will be used.
            context (str?): The context in which the task should be executed.

        Returns:
            str: The output of the task.

        Raises:
            Exception: If the task has no agent assigned, it cannot be executed directly and should be executed in a Crew using a specific process that supports it, like hierarchical.
        """

        agent = agent or self.agent
        if not agent:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, like hierarchical."
            )

        if self.context:
            context = "\n".join([task.output.result for task in self.context])

        result = self.agent.execute_task(
            task=self._prompt(), context=context, tools=self.tools
        )

        self.output = TaskOutput(description=self.description, result=result)
        self.callback(self.output) if self.callback else None
        return result

    def _prompt(self) -> str:
        """        Prompt the task.

        Returns:
            str: Prompt of the task.
                This method generates a prompt for the task. It concatenates the description and expected output (if present) and returns the combined prompt as a string.
        """
        tasks_slices = [self.description]

        if self.expected_output:
            output = self.i18n.slice("expected_output").format(
                expected_output=self.expected_output
            )
            tasks_slices = [self.description, output]
        return "\n".join(tasks_slices)
