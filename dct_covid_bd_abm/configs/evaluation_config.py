import sys

from itertools import product

import numpy as np

from i2mb.utils import global_time

if __name__ != "builtins":
    """If the evaluation configuration is imported as a module, then, we create config instance. If instead, 
    base configuration is executed as part of the configuration creation, we use the instance created externally."""
    from i2mb.engine.configuration import Configuration
    config = Configuration(population_size=0)


def get_night_out_label(night_out_):
    return night_out_ and "bnr" or "no_bnr"


def get_test_to_exit_label(test_to_exit_):
    return test_to_exit_ and "rnt" or "no_rnt"


def get_quarantine_household_label(quarantine_household_):
    return quarantine_household_ and "qch" or "no_qch"


# Stage 1
configurations = [

    dict(name="stage_0",
         sim_steps=200 * global_time.ticks_day,
         experiment_name="no_pathogen_baseline",
         sim_engine=dict(use_pathogen=False, intervene=False, trace_contacts=None, test_to_exit=False,
                         quarantine_household=False),
         scenario=dict(name="apartment_homes"),
         ),

    # validation: Using no intervention scenario no DCT nor MCT nor ICT are used
    dict(name="stage_1",
         experiment_name="validation_baseline",
         sim_engine=dict(intervene=False, trace_contacts=None, test_to_exit=False, quarantine_household=False),
         scenario=dict(name="apartment_homes"),
         ),

    # validation: Using basic intervention scenario, no DCT nor MCT, only ICT is used
    dict(name="stage_1",
         experiment_name="validation_with_ict",
         sim_engine=dict(intervene=True, trace_contacts=None, test_to_exit=False, quarantine_household=False),
         scenario=dict(name="apartment_homes"),
         ),

    # validation: Baseline Using ICT and MCT interventions
    dict(name="stage_1",
         experiment_name="validation_with_ict_mct",
         sim_engine=dict(intervene=True, trace_contacts="MCT", test_to_exit=False, quarantine_household=False),
         scenario=dict(name="apartment_homes"),
         ),
]

# Stage 1.b Parametric search to be able to say which improvement could have the best outcome.
# - Compliance: .9, .8, .7*, .6
# - Coverage: .4, .5*,  .6, .7
# - Result Sharing: .6, .7*, .8, .9
# Compliance = 1 - dropout
# compliance_range = np.linspace(.1, 1., num=10, endpoint=True)
dropout_range = np.arange(0., 0.75, 0.1)
coverage_range = np.arange(0.3, 1.05, 0.1)[::-1]
sharing_range = np.arange(0.3, 1.05, 0.1)[::-1]

# Exploring full range of complementary parameters for the RKI provided parameter values.
fixed_do = 0.3  # From RKI experts
fixed_coverage = 0.5  # From RKI experts
fixed_rs = 0.7  # From RKI experts
grid_space = np.array(list(product(dropout_range, coverage_range, sharing_range))).round(1)
sub_space = grid_space[(grid_space[:, 0] == fixed_do) | (grid_space[:, 1] == fixed_coverage) |
                       (grid_space[:, 2] == fixed_rs)]

for do, coverage, rs in sub_space:
    cfg = dict(name="stage_1_grid",
               experiment_name=f"grid_search_{do:0.2}_{coverage:0.2}_{rs:0.2}",
               sim_engine=dict(intervene=True, trace_contacts="both", test_to_exit=False, quarantine_household=False,
                               dc_tracing=dict(coverage=coverage, dropout=do),
                               get_tested=dict(share_test_result=rs)),
               scenario=dict(name="apartment_homes"),
               )

    configurations.append(cfg)

# DCT introduction
for dct_intro in [0.1, 0.3, 0.5]:
    # Find Boundaries
    dct_intro_time = int(global_time.make_time(day=100) * dct_intro)
    cfg = dict(name="stage_1_dct_introduction_time",
               experiment_name=f"introduction_time_{dct_intro:0.2}",
               sim_engine=dict(intervene=True, trace_contacts=True, test_to_exit=False, quarantine_household=False,
                               # Update coverage and do number with results from previous search
                               dc_tracing=dict(coverage=fixed_coverage, dropout=fixed_do, app_activation_time=dct_intro_time),
                               get_tested=dict(share_test_result=fixed_rs)
                               ),
               scenario=dict(name="apartment_homes"),
               )

    configurations.append(cfg)

# Stage 2
# Prelude best case scenario:
parameters = [
    ["both", False, False, False],
    ["both", False, False, True],  # ("Both", "No Rnt", "No Qch", "Bnr")
    ["both", False, True, True],
    ["both", True, False, False],  # ("Both", "Rnt", "No Qch", "No Bnr")
    ["both", True, False, True],
    ["both", True, True, False],  # ("Both", "Rnt", "No Qch", "No Bnr")
]
for trace_contacts, test_to_exit, quarantine_household, night_out in parameters:
    # Find Boundaries
    e_name = f"{str(trace_contacts).lower()}_{get_test_to_exit_label(test_to_exit)}_" \
             f"{get_quarantine_household_label(quarantine_household)}_{get_night_out_label(night_out)}"

    cfg = dict(name="stage_2_perfect_behaviour",
               experiment_name=f"{e_name}",
               sim_engine=dict(intervene=True, trace_contacts=trace_contacts, test_to_exit=test_to_exit,
                               quarantine_household=quarantine_household, night_out=night_out,
                               # Update coverage and do number with results from previous search
                               dc_tracing=dict(coverage=1, dropout=0,
                                               app_activation_time=0),
                               mc_tracing=dict(dropout=0),
                               get_tested=dict(share_test_result=1)
                               ),
               scenario=dict(name="apartment_homes"),
               )

    configurations.append(cfg)

# Using the DCT parameters found in Stage 1.
# No Intervention baseline Lockdown alone
# print("Remember to set coverage, dropout, and dct_intro to optimal values")
parameters = [True, False]
fixed_dct_intro = int(global_time.make_time(day=100) * 0.1)  # DCT app activated after 10 days
for night_out in parameters:
    # Find Boundaries
    e_name = get_night_out_label(night_out)
    cfg = dict(name="stage_2_night_out",
               experiment_name=f"night_out_{e_name.lower()}",
               sim_engine=dict(intervene=False, trace_contacts=False, test_to_exit=False,
                               quarantine_household=False, night_out=night_out,
                               ),
               scenario=dict(name="apartment_homes"),
               )

    configurations.append(cfg)

# best parameter combination
parameters = product(["both", "MCT"], [True, False], [True, False], [True, False])
for trace_contacts, test_to_exit, quarantine_household, night_out in parameters:
    # Find Boundaries
    e_name = f"{str(trace_contacts).lower()}_{get_test_to_exit_label(test_to_exit)}_" \
             f"{get_quarantine_household_label(quarantine_household)}_{get_night_out_label(night_out)}"

    cfg = dict(name="stage_2_best_main_parameters",
               experiment_name=f"{e_name}",
               sim_engine=dict(intervene=True, trace_contacts=trace_contacts, test_to_exit=test_to_exit,
                               quarantine_household=quarantine_household, night_out=night_out,
                               # Update coverage and do number with results from previous search
                               dc_tracing=dict(coverage=fixed_coverage, dropout=fixed_do,
                                               app_activation_time=fixed_dct_intro),
                               get_tested=dict(share_test_result=fixed_rs)
                               ),
               scenario=dict(name="apartment_homes"),
               )

    configurations.append(cfg)

