class Logger:
    def __init__(self, verbose_level=0):
        verbose_level = (
            2 if isinstance(verbose_level, bool) and verbose_level else verbose_level
        )
        self.verbose_level = verbose_level

    def log(self, level, message):
        """Log a message at the specified level.

        This function checks if the logging is enabled based on the current
        verbose level and logs the message if it meets the criteria.
        Args:
            level (str): The level of severity for the log message, can be "debug" or "info".
            message (str): The message to be logged.
        """

        level_map = {"debug": 1, "info": 2}
        if self.verbose_level and level_map.get(level, 0) <= self.verbose_level:
            print(f"[{level.upper()}]: {message}")
