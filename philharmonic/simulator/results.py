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

    # cloud power consumption
    #------------------
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
        energy_total = evaluator.combined_energy(cloud, env, schedule,
                                                 env.temperature,
                                                 locationBased=conf.location_based)
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



