"""A collection of helper functions for generating the results of the
simulation.

"""

import pickle
from datetime import datetime
import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from philharmonic import conf
import philharmonic as ph
from philharmonic.logger import *
from philharmonic.scheduler import evaluator
from philharmonic.utils import output_loc
from philharmonic import Schedule

def pickle_results(schedule):
    schedule.actions.to_pickle(output_loc('schedule.pkl'))

def generate_series_results(cloud, env, schedule, nplots):
    info('\nDynamic results\n---------------')
    # cloud utilisation
    #------------------
    # evaluator.precreate_synth_power(env.start, env.end, cloud.servers)

    if conf.custom_weights is not None:
        util = evaluator.calculate_cloud_utilisation(cloud, env, schedule, 
                                weights=conf.custom_weights, locationBased=conf.location_based)
    else:
        util = evaluator.calculate_cloud_utilisation(cloud, env, schedule, locationBased=conf.location_based)
    info('Utilisation (%)')
    info(str(util * 100))

    if conf.save_util:
        # fill it out to the full frequency
        util_sampled = util.resample(conf.power_freq, fill_method='pad')
        util_sampled.to_pickle(output_loc('util.pkl'))
    #print('- weighted mean per no')
    # weighted_mean(util[util>0])
    #util[util>0].mean().dropna().mean() * 100
    # TODO: maybe weighted mean for non-zero util
    # ax = plt.subplot(nplots, 1, 1)
    # ax.set_title('Utilisation (%)')
    # util.plot(ax=ax)

    # example: running 100 servers for 14 days at full power (200W)
    #   equals max_power = 0.2kW * 100 * 24 * 14 = 6720 kW = 6.72 MW
    #   for the simulation with 200 vms the total power draw is 70.4 kW
    #    (according to parts of utilisation of max_power)
    #   real power draw from simulation: (power / 1000).sum().sum() = 64.27 kW
    #   utilisation is averaged over all locations: 
    #           util_per_loc = util.sum() / len(util)
    #           total_util = util_per_loc.sum() / len(util_per_loc)
    #   energy costs: assuming an average price of 0.03 $/kWh the costs for this 
    #           simulation are about 2 $
    #           results from real simulation with 100 servers: 1.84 $
    #   exact simulation reference: bcu/2016-02-21/172949_*

    # cloud power consumption
    #------------------    
    if conf.location_based:
        power = evaluator.generate_cloud_power_per_location(util, cloud, env, schedule)
        costs = ph.calculate_price_new(power, env.el_prices, transform_to_jouls=False)
    else:
        # TODO: add frequency to this power calculation
        power = evaluator.generate_cloud_power(util)
    if conf.save_power:
        power.to_pickle(output_loc('power.pkl'))
    ax = plt.subplot(nplots, 1, 3)
    ax.set_title('Computational power (W)')
    power.plot(ax=ax)
    energy = ph.joul2kwh(ph.calculate_energy(power))
    # info('\nEnergy (kWh)')
    # info(energy)
    # info(' - total:')
    # info(energy.sum())
    

    # cooling overhead
    #-----------------
    #temperature = inputgen.simple_temperature()
    if env.temperature is not None:
        power_total = evaluator.calculate_cloud_cooling(power, env.temperature)
    else:
        power_total = power
    ax = plt.subplot(nplots, 1, 4)
    ax.set_title('Total power (W)')
    power_total.plot(ax=ax)
    if conf.save_power:
        power_total.to_pickle(output_loc('power_total.pkl'))
    energy_total = ph.joul2kwh(ph.calculate_energy(power_total))
    # info('\nEnergy with cooling (kWh)')
    # info(energy_total)
    # info(' - total:')
    # info(energy_total.sum())

    if conf.show_pm_frequencies:
        # pm frequencies
        info('\nPM frequencies (MHz)')
        pm_freqs = evaluator.calculate_cloud_frequencies(cloud, env, schedule)
        info(pm_freqs)

    # PM avgutilization
    #info('\nPM Avg.Utilization')
    #info(util.mean())

    info('\nMax of Avg. PM Utilization')
    info(util.mean().max())


    # mean utilization
    info('\nAvg.Utilization')
    info(util.mean().mean())

    info('\nMax Utilization')
    info(util.max().max())


def serialise_results(cloud, env, schedule):

    if conf.location_based:
        return get_results_per_location(cloud, env, schedule)


    fig = plt.figure(1)#, figsize=(10, 15))
    fig.subplots_adjust(bottom=0.2, top=0.9, hspace=0.5)

    nplots = 4
    pickle_results(schedule)
    cloud.reset_to_initial()
    info('Simulation timeline\n-------------------')
    evaluator.print_history(cloud, env, schedule)

    # geotemporal inputs
    #-------------------
    ax = plt.subplot(nplots, 1, 1)
    ax.set_title('Electricity prices ($/kWh)')
    env.el_prices.plot(ax=ax)

    if env.temperature is not None:
        ax = plt.subplot(nplots, 1, 2)
        ax.set_title('Temperature (C)')
        env.temperature.plot(ax=ax)

    # dynamic results
    #----------------
    generate_series_results(cloud, env, schedule, nplots)
    if conf.custom_weights is not None:
        energy = evaluator.combined_energy(cloud, env, schedule, 
                                                    weights=conf.custom_weights, 
                                                    locationBased=conf.location_based)
        energy_total = energy
        # energy_total = evaluator.combined_energy(cloud, env, schedule,
        #                                          env.temperature,
        #                                          locationBased=conf.location_based)
    else:
        energy = evaluator.combined_energy(cloud, env, schedule)
        energy_total = evaluator.combined_energy(cloud, env, schedule,
                                                 env.temperature,
                                                 locationBased=conf.location_based)



    # Aggregated results
    #===================
    info('\nAggregated results\n------------------')

    # migration overhead
    #-------------------
    migration_energy, migration_cost = evaluator.calculate_migration_overhead(
        cloud, env, schedule
    )
    info(' - total energy:')
    info(energy_total)
    info('Migration energy (kWh)')
    info(migration_energy)
    info(' - total energy with migrations:')
    info(energy_total + migration_energy)
    info('\nMigration cost ($)')
    info(migration_cost)


    # electricity costs
    #------------------
    # TODO: update the dynamic cost calculations to work on the new power model
    # TODO: reenable
    # en_cost_IT = evaluator.calculate_cloud_cost(power, env.el_prices)
    info('\nElectricity costs ($)')
    # info(' - electricity cost without cooling:')
    # info(en_cost_IT)
    info(' - total electricity cost without cooling:')
    if conf.custom_weights is not None:
        en_cost_IT_total = evaluator.combined_cost(cloud, env, schedule, env.el_prices, 
                                                    weights=conf.custom_weights, 
                                                    locationBased=conf.location_based)
    else:
        en_cost_IT_total = evaluator.combined_cost(cloud, env, schedule,
                                                   env.el_prices, 
                                                   locationBased=conf.location_based)

    info(en_cost_IT_total)

    # TODO: reenable
    # en_cost_with_cooling = evaluator.calculate_cloud_cost(power_total,
    #                                                       env.el_prices)
    # info(' - electricity cost with cooling:')
    # info(en_cost_with_cooling)
    info(' - total electricity cost with cooling:')
    en_cost_with_cooling_total = evaluator.combined_cost(cloud, env, schedule,
                                                         env.el_prices,
                                                         env.temperature)
    if conf.custom_weights is not None:
        en_cost_with_cooling_total = evaluator.combined_cost(cloud, env, schedule, 
                                        env.el_prices, env.temperature, 
                                        weights=conf.custom_weights, 
                                        locationBased=conf.location_based)
    else:
        en_cost_with_cooling_total = evaluator.combined_cost(cloud, env, schedule,
                                        env.el_prices, env.temperature, 
                                        locationBased=conf.location_based)
        
    info(en_cost_with_cooling_total)
    info(' - total electricity cost with migrations:')
    en_cost_combined = en_cost_with_cooling_total + migration_cost
    info(en_cost_combined)

    # the schedule if we did not apply any frequency scaling
    schedule_unscaled = Schedule()
    schedule_unscaled.actions = schedule.actions[
        schedule.actions.apply(lambda a : not a.name.endswith('freq'))
    ]

    # QoS aspects
    info(' - total profit from users:')
    serv_profit = evaluator.calculate_service_profit(cloud, env, schedule)
    info('${}'.format(serv_profit))
    info(' - profit loss due to scaling:')
    serv_profit_unscaled = evaluator.calculate_service_profit(
        cloud, env, schedule_unscaled
    )
    scaling_profit_loss = serv_profit_unscaled - serv_profit
    scaling_profit_loss_rel = scaling_profit_loss / serv_profit_unscaled
    info('${}'.format(scaling_profit_loss))
    info('{:.2%}'.format(scaling_profit_loss_rel))

    # frequency savings
    info(' - frequency scaling savings (compared to no scaling):')
    if conf.custom_weights is not None:
        en_cost_combined_unscaled = evaluator.combined_cost(
            cloud, env, schedule_unscaled, env.el_prices, env.temperature, 
            weights=conf.custom_weights, locationBased=conf.location_based
        ) + migration_cost
    else:
        en_cost_combined_unscaled = evaluator.combined_cost(
            cloud, env, schedule_unscaled, env.el_prices, env.temperature,
            locationBased=conf.location_based
        ) + migration_cost
    scaling_savings_abs = en_cost_combined_unscaled - en_cost_combined
    info('${}'.format(scaling_savings_abs))
    scaling_savings_rel = scaling_savings_abs / en_cost_combined_unscaled
    info('{:.2%}'.format(scaling_savings_rel))

    #------------------
    # Capacity constraints
    #---------------------
    # TODO: these two

    # aggregated results
    aggregated = [energy, en_cost_IT_total,
                  energy_total + migration_energy, en_cost_combined,
                  serv_profit, serv_profit - en_cost_combined]
    aggr_names = ['IT energy (kWh)', 'IT cost ($)',
                  'Total energy (kWh)', 'Total cost ($)',
                  'Service revenue ($)', 'Gross profit ($)']
    # http://en.wikipedia.org/wiki/Gross_profit
    # Towards Profitable Virtual Machine Placement in the Data Center Shi
    # and Hong 2011 - total profit, revenue and operational cost
    aggregated_results = pd.Series(aggregated, aggr_names)
    aggregated_results.to_pickle(output_loc('results.pkl'))
    #aggregated_results.plot(kind='bar')
    info('\n')
    info(aggregated_results)

    if conf.liveplot:
        plt.show()
    elif conf.fileplot:
        plt.savefig(output_loc('results-graph.pdf'))

    info('\nDone. Results saved to: {}'.format(conf.output_folder))

    return aggregated_results


def serialise_results_batch(simulation_parameters):

    info("serialise_results_batch\n-----------------")

    power_vs_costs = {}

    for sim_param in simulation_parameters.items():

        scenario = sim_param[0]
        [ cloud, env, schedule ] = sim_param[1]

        info("====================\n\n")
        info("scenario: "+str(scenario))


        [ total_power, total_costs ] = get_results_per_location(cloud, env, schedule)

        power_vs_costs[scenario] = [ total_power, total_costs ]


    max_power = max(power_vs_costs.items(), key=lambda x: x[1][0])[1][0]
    max_cost = max(power_vs_costs.items(), key=lambda x: x[1][1])[1][1]

    power_values = [ item[1][0] for item in power_vs_costs.items() ]
    cost_values = [ item[1][1] for item in power_vs_costs.items() ]

    norm_power_values = [ p / max_power for p in power_values ]
    norm_cost_values = [ c / max_cost for c in cost_values ]

    info("power values")
    info(power_values)
    info("cost values")
    info(cost_values)
    info("norm power values")
    info(norm_power_values)
    info("norm cost values")
    info(norm_cost_values)


def get_results_per_location(cloud, env, schedule):
    """go through the given schedule and generate results
    calculate metrics per location
    """

    info("------------------")
    info("serialise results")
    info("------------------")

    # output
    fig = plt.figure(1)#, figsize=(10, 15))
    fig.subplots_adjust(bottom=0.2, top=0.9, hspace=0.5)

    nplots = 4
    pickle_results(schedule)
    cloud.reset_to_initial()
    info('Simulation timeline\n-------------------')

    # geotemporal inputs
    #-------------------
    ax = plt.subplot(nplots, 1, 1)
    ax.set_title('Electricity prices ($/kWh)')
    env.el_prices.plot(ax=ax)


    # # calculate utilisation
    # if conf.custom_weights is not None:
    #     util = evaluator.calculate_cloud_utilisation(cloud, env, schedule, 
    #                             weights=conf.custom_weights, locationBased=conf.location_based)
    # else:
    #     util = evaluator.calculate_cloud_utilisation(cloud, env, schedule,
    #                             locationBased=conf.location_based)

    # migration_energy, migration_cost = evaluator.calculate_custom_migration_overhead(
    #     cloud, env, schedule, bandwidth_map=conf.bandwidth_map
    # )

    # [ total_penalty_cost, total_downtime, num_migrations ] = evaluator.calculate_custom_sla_penalties(cloud, env, schedule)

    if conf.custom_weights is not None:

        [  cloud_util, active_servers, migration_energy, migration_cost,
            total_penalty_cost, total_downtime, num_migrations ] = evaluator.calculate_cloud_metrics(cloud, env, schedule, weights=conf.custom_weights, bandwidth_map=conf.bandwidth_map)

    else:

        [  cloud_util, active_servers, migration_energy, migration_cost,
            total_penalty_cost, total_downtime, num_migrations ] = evaluator.calculate_cloud_metrics(cloud, env, schedule, bandwidth_map=conf.bandwidth_map)

    # calculate cloud power in kWh
    cloud_power = evaluator.generate_cloud_power_per_location(cloud_util, active_servers, cloud, env, schedule)
    # cloud costs per location
    cloud_costs = ph.calculate_price_new(cloud_power, env.el_prices, transform_to_jouls=False)

    total_cloud_power = cloud_power.sum().sum() # in kWh
    total_cloud_costs = cloud_costs.sum() # in $

    total_power = total_cloud_power + migration_energy
    total_costs = total_cloud_costs + migration_cost + total_penalty_cost

    info('Utilisation (%)')
    info(str(cloud_util * 100))

    info('\nMax of Avg. PM Utilization')
    info(cloud_util.mean().max())

    # mean utilization
    info('\nAvg.Utilization')
    info(cloud_util.mean().mean())

    info('\nMax Utilization')
    info(cloud_util.max().max())

    # Aggregated results
    #===================
    info('\nAggregated results\n------------------')
    info(' - cloud power (kWh)')
    info(total_cloud_power)
    info(' - cloud cost ($)')
    info(total_cloud_costs)
    info('Migration energy (kWh)')
    info(migration_energy)
    info('Migration cost ($)')
    info(migration_cost)
    info(' - cloud power with migrations:')
    info(total_cloud_power + migration_energy)
    info(' - cloud cost with migrations:')
    info(total_cloud_costs + migration_cost)
    info(' - total penalty cost')
    info(total_penalty_cost)
    info(' - total downtime (s)')
    info(total_downtime)
    info(' - total number of migrations')
    info(num_migrations)
    info(' - total power (kWh)')
    info(total_power)
    info(' - total cost ($)')
    info(total_costs)

    if conf.liveplot:
        plt.show()
    elif conf.fileplot:
        plt.savefig(output_loc('results-graph.pdf'))

    info('\nDone. Results saved to: {}'.format(conf.output_folder))
    
    return [ total_power, total_costs ]


def serialise_results_tests():
    def update_line(num, data, line):
        line.set_data(data[...,:num])
        return line,

    fig1 = plt.figure()

    data = np.random.rand(2, 25)
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.title('test')
    line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
        interval=50, blit=True)
    #line_ani.save('lines.mp4')

    fig2 = plt.figure()

    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    base = np.hypot(x, y)
    ims = []
    for add in np.arange(15):
        ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
        blit=True)
    #im_ani.save('im.mp4', metadata={'artist':'Guido'})

    plt.show()



