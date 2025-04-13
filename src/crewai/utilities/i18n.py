import json
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator


class I18N(BaseModel):
    _translations: Dict[str, Dict[str, str]] = PrivateAttr()
    language: Optional[str] = Field(
        default="en",
        description="Language used to load translations",
    )

    @model_validator(mode="after")
    def load_translation(self) -> "I18N":
        """Load translations from a JSON file based on the specified language.

        This function attempts to load translation data for the given language
        from a JSON file. It constructs the path to the JSON file using the
        language code and reads the file content. If the file is not found, it
        raises a `ValidationError` indicating that the translation file for the
        specified language does not exist. Similarly, if there is an error
        decoding the JSON content, another `ValidationError` is raised.

        Returns:
            I18N: The current instance of the class with translations loaded.

        Raises:
            ValidationError: If the translation file for the specified language is not found.
            ValidationError: If there is an error decoding the JSON content from the prompts file.
        """
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            prompts_path = os.path.join(
                dir_path, f"../translations/{self.language}.json"
            )

            with open(prompts_path, "r") as f:
                self._translations = json.load(f)
        except FileNotFoundError:
            raise ValidationError(
                f"Translation file for language '{self.language}' not found."
            )
        except json.JSONDecodeError:
            raise ValidationError(f"Error decoding JSON from the prompts file.")

        if not self._translations:
            self._translations = {}

        return self

    def slice(self, slice: str) -> str:
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        return self.retrieve("errors", error)

    def tools(self, error: str) -> str:
        return self.retrieve("tools", error)

    def retrieve(self, kind, key) -> str:
        """Retrieve a translation for a given kind and key.

        Args:
            kind (str): The category or type of the translation.
            key (str): The specific key within the category to retrieve the translation for.

        Returns:
            str: The translation corresponding to the provided kind and key.

        Raises:
            ValidationError: If no translation is found for the specified kind and key.
        """

        try:
            return self._translations[kind][key]
        except:
            raise ValidationError(f"Translation for '{kind}':'{key}'  not found.")
