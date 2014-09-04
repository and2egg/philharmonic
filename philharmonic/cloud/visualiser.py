'''
Created on Sep 4, 2014

@author: kermit

Functions for printing and visualising model data

'''

from philharmonic.logger import info, debug

get_locations = lambda cloud : list(set([s.loc for s in cloud.servers]))
servers_at_location = lambda cloud, loc : [s for s in cloud.servers if s.loc == loc]

def create_usage_str(usage, bins=8):
    sum_ = 0
    filled = 0
    while usage > sum_:
        sum_ += 1. / bins
        filled += 1
    if filled > bins:
        filled = bins
    return filled * '#' + (bins - filled) * '_'

def server_usage_repr(server_free, server_cap):
    server_usage_str = ''
    for res, res_free in server_free.iteritems():
        res_usage = 1 - res_free / float(server_cap[res])
        res_usage_str = create_usage_str(res_usage)
        server_usage_str += '{}:|{}| '.format(res, res_usage_str)
    return server_usage_str

# TODO: make this work for state and cloud.show_usage() call it with get_current
def show_usage(cloud):
    locations = get_locations(cloud)
    for location in locations:
        info("\n{}\n----------".format(location))
        servers = servers_at_location(cloud, location)
        for server in servers:
            server_str = server.full_info()
            server_free = cloud.get_current().free_cap[server]
            server_usage = server_usage_repr(server_free, server.cap)
            vms = list(cloud.get_current().alloc[server])
            #vms_string = str([vm for vm in vms])
            vms_string = str([vm.full_info() for vm in vms])
            #info("{}\n - {}\n - {}".format(server, server_usage, vms))
            info("{} - {} - {}".format(server_usage, server_str, vms_string))
