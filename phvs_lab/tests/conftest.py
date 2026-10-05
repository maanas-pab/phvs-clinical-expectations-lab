import pytest

from phvs_lab.modules import data_loader as dl
from phvs_lab.modules import valuation as val


@pytest.fixture(scope="session")
def trials():
    return dl.get_clinical_trials()


@pytest.fixture(scope="session")
def assumptions():
    return dl.get_commercial_assumptions()


@pytest.fixture(scope="session")
def base_params():
    return val.params_from_assumptions()
