import pytest

def pytest_addoption(parser):
    parser.addoption("--save_results", action="store_true", help="Flag to save test folder")
    parser.addoption("--slurm_mode", action="store_true", help="Flag to run by submit SLURM jobs, else run on interactive node")
    parser.addoption("--filesys_prefix", action="store", default="/tmp", help="Temp path to run tests")

@pytest.fixture(scope="module")
def get_options(request):
    options_list = ("save_results", "slurm_mode", "filesys_prefix")
    user_options = {option : request.config.getoption("--" + option) for option in options_list}
    return user_options
