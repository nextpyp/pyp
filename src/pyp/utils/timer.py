"""Timer module adapted from `Real Python`_.


.. _Real Python:
    https://realpython.com/python-timer/
"""

import functools
import time
import datetime
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Optional

__author__ = "Xiaochen Du"
__maintainer__ = "Alberto Bartesaghi"
__email__ = "alberto@cs.duke.edu"

DEFAULT_TEXT = "Elapsed time: {}"


class TimerError(Exception):
    """A custom exception used to report errors in the use of Timer class."""


@dataclass
class Timer(ContextDecorator):
    timers: ClassVar[Dict[str, Dict[str, str]]] = dict()
    name: Optional[str] = None
    text: str = DEFAULT_TEXT
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialization: Add new named timers to dictionary of timers"""
        if self.name:
            self.timers.setdefault(self.name, {"elapsed_time": 0, "start_time": "0", "end_time": "0"})

    def start(self):
        """Start a new timer"""
        
        self.timers[self.name] = {"elapsed_time": 0, "start_time": str(datetime.datetime.now()), "end_time": "N/A"}
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            days = elapsed_time // (24 * 3600)
            timed = elapsed_time % (24 * 3600)
            hours = timed // 3600
            timed %= 3600
            minutes = timed // 60
            timed %= 60
            seconds = timed
            if days > 0:
                self.logger(self.text.format("%dd %dh %dm %ds" % (days, hours, minutes, seconds)))
            elif hours > 0:
                self.logger(self.text.format("%dh %dm %ds" % (hours, minutes, seconds)))
            elif minutes > 0:
                self.logger(self.text.format("%dm %ds" % (minutes, seconds)))
            else:
                self.logger(self.text.format("%ds" % (seconds)))
                
        if self.name:
            self.timers[self.name]["elapsed_time"] += elapsed_time
            self.timers[self.name]["end_time"] = str(datetime.datetime.now())

        return elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()

    def __exit__(self, *exc_info):
        """Stop the context manager"""
        self.stop()

    def __call__(self, func):
        """Support using Timer as a decorator"""

        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper_timer
