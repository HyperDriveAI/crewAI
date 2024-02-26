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
        """        Load translations from a JSON file based on the specified language.

        This method loads translations from a JSON file based on the specified language. It first constructs the path to the
        translations file using the specified language, then attempts to open and read the file. If successful, it loads the
        translations into the '_translations' attribute. If the file is not found, a 'ValidationError' is raised with a
        message indicating the missing translation file. If there is an error decoding the JSON from the file, a
        'ValidationError' is raised with a corresponding error message. Finally, if no translations are loaded, an empty
        dictionary is assigned to the '_translations' attribute.

        Returns:
            I18N: The current instance of the 'I18N' class.

        Raises:
            ValidationError: If the translation file for the specified language is not found or if there is an error
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
        """        Retrieve a specific slice from the database.

        Args:
            slice (str): The name of the slice to retrieve.

        Returns:
            str: The retrieved slice.
        """

        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        """        Retrieve the specified error message from the 'errors' collection.

        Args:
            error (str): The error message to retrieve.

        Returns:
            str: The retrieved error message.
        """

        return self.retrieve("errors", error)

    def tools(self, error: str) -> str:
        """        Retrieve tools based on the given error.

        This function retrieves tools based on the provided error. It calls the 'retrieve' method with the parameters
        'tools' and the given error to fetch the required tools.

        Args:
            error (str): The error for which tools need to be retrieved.

        Returns:
            str: The retrieved tools based on the given error.
        """

        return self.retrieve("tools", error)

    def retrieve(self, kind, key) -> str:
        """        Retrieve the translation for the given kind and key.

        Args:
            kind (str): The category of the translation.
            key (str): The specific key for the translation.

        Returns:
            str: The translation corresponding to the given kind and key.

        Raises:
            ValidationError: If the translation for the given kind and key is not found.
        """

        try:
            return self._translations[kind][key]
        except:
            raise ValidationError(f"Translation for '{kind}':'{key}'  not found.")
