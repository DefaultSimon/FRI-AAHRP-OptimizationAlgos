import time
from typing import Optional, Callable, Dict


class Timer:
    """
    Timer is a utility class meant to make measuring time deltas easier.

    Usage with automatic print:
    >>> with Timer("This took {delta} seconds.", print_on_context_exit=True):
    >>>     time.sleep(5)
    "This took 5 seconds."

    Manual (more common) usage:
    >>> timer = Timer()
    >>> with timer:
    >>>     time.sleep(10)
    >>> print(timer.get_delta())
    10.0

    Or:
    >>> timer = Timer()
    >>> timer.start()
    >>> time.sleep(10)
    >>> timer.end()
    >>> print(timer.get_delta())
    10.0
    """
    def __init__(
        self,
        print_message: Optional[str] = None,
        print_decimal_places: int = 2,
        print_on_context_exit: bool = False,
        logging_function: Callable = print,
    ):
        """
        Instantiate a new Timer.

        :param print_message: String that will be formatted upon calling print_delta.
            Must include the placeholder "{delta}" which will be replaced when printing.
        :param print_decimal_places: How many decimal places to format the number to.
        :param logging_function: Which function to use for printing, defaults to log.info.
        :param print_on_context_exit: Whether to print the delta (using the specified message)
            upon exiting a context (see example above).
        """
        self._message: Optional[str] = print_message
        self._decimal_places: int = print_decimal_places
        self._print_on_context_exit: bool = print_on_context_exit

        self._time_start: Optional[float] = None
        self._time_end: Optional[float] = None

        self._log_fn: Callable = logging_function

    def _ensure_done(self):
        if self._time_start is None:
            raise RuntimeError("start_timer was never called!")
        if self._time_end is None:
            raise RuntimeError("end_timer was never called!")

    def start(self):
        """
        Starts (or resets) the timer.
        """
        self._time_start = time.time()

    def end(self, print_delta: bool = False):
        """
        Stops the timer and optionally prints the delta.

        :param print_delta: Whether to print the delta when calling this method.
        """
        self._time_end = time.time()

        if print_delta:
            self.print_delta()

    def reset(self) -> None:
        """
        Reset the timer.
        """
        self._time_start = None
        self._time_end = None

    def get_delta(self) -> float:
        """
        Return the time delta between starting and stopping the timer.
        """
        self._ensure_done()

        return self._time_end - self._time_start

    def get_time_since_start(self) -> float:
        """
        Return the time delta between the starting point and this moment (does not stop the actual timer).
        """
        return time.time() - self._time_start

    def print_delta(self, format_kwargs: Optional[Dict[str, str]] = None):
        """
        Print the delta (seconds) that have passed between starting and stopping the timer.

        :param format_kwargs: Additional format keys and values to pass when formatting the message.
        """
        self._ensure_done()
        if not self._message:
            raise ValueError("No message set, can't print!")

        time_delta: float = round(self._time_end - self._time_start, self._decimal_places)

        if format_kwargs is not None:
            self._log_fn(self._message.format(delta=time_delta, **format_kwargs))
        else:
            self._log_fn(self._message.format(delta=time_delta))

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end(print_delta=self._print_on_context_exit)
