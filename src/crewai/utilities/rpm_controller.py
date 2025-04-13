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
        """Reset the counter associated with the RPM limit.

        This method resets the request count when a maximum RPM (Revolutions Per
        Minute) is set. It ensures that the counter is properly locked before
        resetting to avoid race conditions.

        Args:
            self: The instance of the object on which this method is called.

        Returns:
            self: Returns the current instance, allowing for method chaining.
        """

        if self.max_rpm:
            self._lock = threading.Lock()
            self._reset_request_count()
        return self

    def check_or_wait(self):
        """Check if the current RPM is below the maximum RPM and increment it if
        possible.

        If the `max_rpm` attribute is not set, the method returns `True`.
        Otherwise, it checks if the current RPM (`_current_rpm`) is less than
        the maximum allowed RPM (`max_rpm`). If so, it increments `_current_rpm`
        by 1 and returns `True`. If the current RPM equals or exceeds the
        maximum RPM, it logs an informational message indicating that the max
        RPM has been reached. The method then waits for the next minute using
        `_wait_for_next_minute()` and resets `_current_rpm` to 1 before
        returning `True`.

        Returns:
            bool: Always returns `True`.
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
        """Stop the RPM counter.

        If the `_timer` attribute is not `None`, it cancels the timer and sets
        it to `None`.
        """

        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _wait_for_next_minute(self):
        """Wait until the next minute and reset the current RPM to zero.

        This function uses a lock to ensure thread safety when resetting the RPM
        value. It pauses the execution for one minute using time.sleep(60) and
        then acquires the lock, setting the _current_rpm attribute to 0.
        """

        time.sleep(60)
        with self._lock:
            self._current_rpm = 0

    def _reset_request_count(self):
        """Reset the request count to zero and restart the timer.

        This function resets the current request per minute (RPM) counter to
        zero and cancels any existing timer. It then schedules a new timer that
        will call itself again after 60 seconds, effectively resetting the RPM
        count every minute.
        """

        with self._lock:
            self._current_rpm = 0
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(60.0, self._reset_request_count)
        self._timer.start()
