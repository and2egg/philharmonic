from nose.tools import *
from mock import MagicMock, patch
import pandas as pd

from philharmonic import Schedule
from philharmonic.scheduler.bcffs_scheduler import *

from philharmonic.simulator.environment import FBFSimpleSimulatedEnvironment
from philharmonic.simulator.simulator import FBFSimulator
from philharmonic.simulator import inputgen
from philharmonic import Cloud, VMRequest, \
    VM, Server, State, Migration

@patch('philharmonic.scheduler.BCFFSScheduler._schedule_frequency_scaling')
def test_bcffs_returns_schedule(mock_schedule_frequency_scaling):
    mock_schedule_frequency_scaling = MagicMock(return_value = None)
    scheduler = BCFFSScheduler()
    scheduler.cloud = Cloud()
    scheduler.environment = FBFSimpleSimulatedEnvironment()
    scheduler.environment.get_requests = MagicMock(return_value = [])
    scheduler.environment.get_time = MagicMock(return_value = 1)
    schedule = scheduler.reevaluate()
    assert_is_instance(schedule, Schedule)

@patch('philharmonic.scheduler.BCFFSScheduler._schedule_frequency_scaling')
def test_bcf_reevaluate_underutilised(mock_schedule_frequency_scaling):
    mock_schedule_frequency_scaling = MagicMock(return_value = None)
    scheduler = BCFFSScheduler()
    scheduler.environment = FBFSimpleSimulatedEnvironment()
    s1, s2 = Server(40000, 12, location='A'), Server(20000, 10, location='A')
    vm1 = VM(2000, 1)
    vm2 = VM(2000, 1)
    r2 = VMRequest(vm2, 'boot')
    cloud = Cloud([s1, s2], [vm1, vm2])
    scheduler.cloud = cloud
    scheduler.environment.get_requests = MagicMock(return_value = [r2])
    scheduler.environment.t = 1
    el = pd.DataFrame({'A': [0.08], 'B': [0.08]}, [1])
    temp = pd.DataFrame({'A': [15], 'B': [15]}, [1])
    scheduler.environment.current_data = MagicMock(return_value = (el, temp))
    # the initial state is a VM hosted on an underutilised PM
    cloud.apply_real(Migration(vm1, s2))
    schedule = scheduler.reevaluate()
    for action in schedule.actions:
        cloud.apply_real(action)
    current = cloud.get_current()
    assert_true(current.all_within_capacity())
    assert_equals(current.allocation(vm1), s1)
    assert_equals(current.allocation(vm2), s1)
    assert_true(current.all_allocated())

@patch('philharmonic.scheduler.evaluator.conf')
def test_bcf_reevaluate_freq_scaling(mock_conf):
    f_max = 3000
    mock_conf.f_max = f_max
    mock_conf.power_freq_model = True
    mock_conf.P_idle = 100
    mock_conf.P_std = 5
    mock_conf.P_dif = 15
    mock_conf.P_base = 150
    mock_conf.power_freq = '5min'
    mock_conf.pricing_freq = '1h'

    scheduler = BCFFSScheduler()
    times = pd.date_range('2013-02-25 00:00', periods=48, freq='H')
    scheduler.environment = FBFSimpleSimulatedEnvironment(times)
    s1, s2 = Server(40000, 12, location='A'), Server(20000, 10, location='A')
    vm1 = VM(2000, 1)
    vm2 = VM(2000, 1)
    r2 = VMRequest(vm2, 'boot')
    cloud = Cloud([s1, s2], [vm1, vm2])
    scheduler.cloud = cloud
    scheduler.environment.get_requests = MagicMock(return_value = [r2])
    t = times[0]
    el = pd.DataFrame({'A': [0.08] * len(times),
                       'B': [0.08] * len(times)}, times)
    temp = pd.DataFrame({'A': [15] * len(times), 'B': [15] * len(times)}, times)
    scheduler.environment.current_data = MagicMock(return_value = (el, temp))
    # the initial state is a VM hosted on an underutilised PM
    cloud.apply_real(Migration(vm1, s2))
    schedule = scheduler.reevaluate()
    for action in schedule.actions:
        cloud.apply_real(action)
    current = cloud.get_current()
    assert_true(current.all_within_capacity())
    assert_equals(current.allocation(vm1), s1)
    assert_equals(current.allocation(vm2), s1)
    assert_true(current.all_allocated())