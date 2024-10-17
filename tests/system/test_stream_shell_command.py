
from pyp.system.local_run import stream_shell_command


class Logger:

    def __init__(self):
        self.f = lambda line: self.log(line)
        self.lines = []

    def log(self, line):
        self.lines.append(line)


def test_cmd():

    logger = Logger()
    proc = stream_shell_command('echo foo', log=logger.f)

    assert proc.returncode == 0
    assert logger.lines == ['foo']


def test_observer():

    logger = Logger()
    observed_lines = []

    def obs(line):
        observed_lines.append(line)

    proc = stream_shell_command('echo foo', log=logger.f, observer=obs)

    assert proc.returncode == 0
    assert observed_lines == ['foo']


def test_observer_stop():

    logger = Logger()
    observed_lines = []

    def obs(line):
        observed_lines.append(line)
        if line == 'foo':
            return False

    proc = stream_shell_command('echo foo; sleep 0.2; echo bar;', log=logger.f, observer=obs)

    assert proc.returncode == -15  # SIGTERM
    assert logger.lines == ['foo']
    assert observed_lines == ['foo']
