"""generate artificial input"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta

from philharmonic import Machine, Server, VM, VMRequest, Cloud

# Cummon functionality
#---------------------

def normal_population(num, bottom, top, ceil=True):
    """ Return array @ normal distribution.
    bottom, top are approx. min/max values.

    """
    half = (top - bottom)/2.0
    # we want 99% of the population to enter the [min,max] interval
    sigma = half/3.0
    mu = bottom + half
    print(mu, sigma)

    values = mu + sigma * np.random.randn(num)
    # negative to zero
    values[values<0]=0
    if ceil:
        values = np.ceil(values).astype(int)
    return values


# DC description
#---------------

def small_infrastructure():
    """return a list of Servers with determined resource capacities"""
    num_servers = 3
    Machine.resource_types = ['RAM', '#CPUs']
    RAM = [4]*num_servers
    numCPUs = [2]*num_servers
    servers = []
    for i in range(num_servers):
        s = Server(RAM[i], numCPUs[i])
        servers.append(s)
    return servers

def peak_pauser_infrastructure():
    """1 server hosting 1 vm"""
    server = Server()
    vm = VM()
    cloud = Cloud([server], [vm])
    return cloud

# VM requests
#------------
# - global settings TODO: config file
VM_num = 7
# e.g. CPUs
min_size = 1
max_size = 8
# e.g. seconds
min_duration = 60 * 60 # 1 hour
max_duration = 60 * 60 * 3 # 3 hours
#max_duration = 60 * 60 * 24 * 10 # 10 days

def normal_vmreqs(start, end=None):
    """Generate the VM creation and deletion events in. 
    Normally distributed arrays - VM sizes and durations.
    @param start, end - time interval (events within it)

    """
    delta = end - start
    # array of VM sizes
    sizes = normal_population(VM_num, min_size, max_size)
    # duration of VMs
    durations = normal_population(VM_num, min_duration, max_duration)
    requests = []
    moments = []
    for size, duration in zip(sizes, durations):
        vm = VM(size)
        # the moment a VM is created
        offset = pd.offsets.Second(np.random.uniform(0., delta.total_seconds()))
        requests.append(VMRequest(vm, 'boot'))
        moments.append(start + offset)
        # the moment a VM is destroyed
        offset += pd.offsets.Second(duration)
        if start + offset <= end: # event is relevant
            requests.append(VMRequest(vm, 'delete'))
            moments.append(start + offset)
    events = pd.TimeSeries(data=requests, index=moments)
    return events.sort_index()
