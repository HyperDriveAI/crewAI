from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    result: str = Field(description="Result of the task")

    @model_validator(mode="after")
    def set_summary(self):
        """Set a summary of the description.

        Extracts the first 10 words from the description and appends an ellipsis
        to create a concise summary.

        Returns:
            object: The current object with the summary set.
        """

        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self
