import threading
import time
from typing import Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from crewai.utilities.logger import Logger


class RPMController(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    max_rpm: Union[int, None] = Field(default=None)
    logger: Logger = Field(default=None)
    _current_rpm: int = PrivateAttr(default=0)
    _timer: threading.Timer | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default=None)

    @model_validator(mode="after")
    def reset_counter(self):
        """        Reset the counter and lock if the maximum RPM is set.

        This method checks if the maximum RPM is set, and if so, it resets the counter and initializes a threading lock.

        Returns:
            The current instance of the object.
        """

        if self.max_rpm:
            self._lock = threading.Lock()
            self._reset_request_count()
        return self

    def check_or_wait(self):
        """        Check if the current RPM is less than the maximum RPM. If it is, increment the current RPM by 1.
        If the current RPM is equal to the maximum RPM, log a message, wait for the next minute to start,
        and reset the current RPM to 1.

        Returns:
            bool: True if the current RPM is less than the maximum RPM or if the current RPM has been incremented/reset;
            otherwise, False.
        """

        if not self.max_rpm:
            return True

        with self._lock:
            if self._current_rpm < self.max_rpm:
                self._current_rpm += 1
                return True
            else:
                self.logger.log(
                    "info", "Max RPM reached, waiting for next minute to start."
                )
                self._wait_for_next_minute()
                self._current_rpm = 1
                return True

    def stop_rpm_counter(self):
        """        Stop the RPM counter timer.

        This function stops the RPM counter timer if it is running.
        """

        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        """        Wait for the next minute and reset the current RPM to 0.

        This function pauses the program execution for 60 seconds using time.sleep(60) and then resets the current RPM to 0
        by acquiring a lock and updating the _current_rpm attribute.
        """

        time.sleep(60)
        with self._lock:
            self._current_rpm = 0

    def _reset_request_count(self):
        """        Reset the request count and start a new timer.

        This function resets the request count to 0 and starts a new timer to reset the count after 60 seconds.
        """

        with self._lock:
            self._current_rpm = 0
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(60.0, self._reset_request_count)
        self._timer.start()
