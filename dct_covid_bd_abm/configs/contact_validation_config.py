import numpy as np

from i2mb.utils import global_time

if __name__ != "builtins":
    """If the evaluation configuration is imported as a module, then, we create config instance. If instead, 
    base configuration is executed as part of the configuration creation, we use the instance created externally."""
    from i2mb.engine.configuration import Configuration
    config = Configuration()

configurations = [
    dict(
        name="contact_validation",
        experiment_name="POLYMOD",
        sim_engine=dict(intervene=False, trace_contacts=None,
                        test_to_exit=False,
                        quarantine_household=False,
                        night_out=True,
                        use_pathogen=False,
                        use_contact_history=True,
                        activity_manager=dict(write_diary=True)
                        )
    )
]
covimod_waves = ["COVIMOD 1", "COVIMOD 2", "COVIMOD 3", "COVIMOD 4"]
goes_to_work_percentage = [.05, .20, .40, .60]
bnr_open = [False, False, True, True]

# Generate CVIMOD configuration
for replication_experiment, gtw_percentage, bnr in zip(covimod_waves, goes_to_work_percentage, bnr_open ):
    cfg = dict(
        name="contact_validation",
        experiment_name=f"{replication_experiment}",
        sim_engine=dict(intervene=True, trace_contacts="MCT",
                        test_to_exit=False,
                        quarantine_household=False,
                        night_out=bnr,
                        daily_schedule=dict(must_work=.1,
                                            stays_home=0.9 - gtw_percentage),
                        night_out_module=dict(
                                group_location=["home", "office"],
                                min_capacity=0.5,
                                opening_hours=global_time.make_time(hour=17),
                                closing_hours=global_time.make_time(hour=22)
                                )
                        ),
        )
    configurations.append(cfg)
