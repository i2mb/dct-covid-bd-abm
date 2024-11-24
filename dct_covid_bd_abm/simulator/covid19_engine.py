#  dct_mct_analysis
#  Copyright (c) 2021 FAU - RKI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from copy import deepcopy
from functools import partial

import numpy as np

from i2mb.activities.controllers.default_activity_controller import DefaultActivityController
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.controllers.location_activities import LocationActivitiesController
from i2mb.activities.controllers.sleep_controller import SleepBehaviourController
from i2mb.activities.night_out import NightOut
from i2mb.activities.schedule_routines import ScheduleRoutines, Schedule
from i2mb.behaviours.get_tested import GetTested
from i2mb.engine.core import Engine
from i2mb.interactions.contact_history import ContactHistory
from i2mb.interactions.digital_contact_tracing import RegionContactTracing
from i2mb.interactions.fnf_contact_tracing import FriendsNFamilyContactTracing
from i2mb.interactions.manual_contact_tracing import ManualContactTracing
from i2mb.interventions.contact_isolation import ContactIsolationIntervention
from i2mb.interventions.test import Test
from i2mb.motion.ambulance import Ambulance
from i2mb.motion.random_motion import RandomMotion
from i2mb.motion.target_motion import MoveToTarget
from i2mb.motion.undertaker import Undertaker
from i2mb.pathogen import SymptomLevels, RegionVirusDynamicExposure
from i2mb.pathogen.infection import RegionCoronaVirusExposureWindow
from i2mb.utils import global_time


class COVID19ContactTracing:
    def __init__(self, scenario, hospitalize=False, night_out=False, intervene=True,
                 test_to_exit=True,
                 use_buses=False,
                 trace_contacts=True,
                 quarantine_households=False,
                 use_pathogen=True,
                 use_contact_history=False,
                 dropout=.2,
                 sleep=True,
                 base_file_name="./",
                 **module_config):

        self.config = module_config
        self.relocator = scenario.relocator
        self.time_scalar = global_time.time_scalar
        self.base_file_name = base_file_name

        self.population = scenario.population
        self.quarantine_household = quarantine_households
        self.world = scenario.world
        self.motion = RandomMotion(self.population, **module_config.get("motion", {}), )

        self.modules = []
        self.modules.extend(scenario.dynamic_regions())

        # Contact tracing Modules
        self.contact_history = self.create_contact_history_module(use_contact_history, scenario)
        self.fnf_tracing = self.create_fnf_contact_tracing_module(intervene, scenario)
        self.c_tracing = self.create_digital_contact_tracing_module(trace_contacts, dropout=dropout)
        self.m_tracing = self.create_manual_contact_tracing_module(trace_contacts, scenario)

        # Pathogen
        self.pathogen = self.create_pathogen_model(use_pathogen)

        # Intervention
        intervention_modules = self.create_intervention_modules(intervene, quarantine_households,
                                                                test_to_exit)
        self.ct_intervention, self.get_tested, self.tester = intervention_modules

        # City management modules
        self.undertaker = Undertaker(self.population, self.relocator, scenario.graveyard)
        self.ambulance = self.create_ambulance_module(hospitalize, scenario)

        # Behaviours
        self.daily_schedule = self.create_daily_schedule_module(use_buses, scenario)
        self.night_out = self.create_night_out_module(night_out, scenario)
        self.activity_manager = self.create_activity_manager()
        self.sleep_model = self.create_sleep_module(sleep)
        self.default_activity_controller = self.create_default_activity_controller()
        self.local_activity_model = self.create_local_activity_module()

        self.register_modules()

        self.engine = Engine(self.modules, base_file_name=base_file_name, debug=True)
        self.post_init_modules()

    def create_sleep_module(self, sleep):
        population = self.population
        sleep_module = None
        if sleep:
            sleep_model_cfg = self.config.get("sleep_model", {})
            sleep_module = SleepBehaviourController(population,
                                                    activity_manager=self.activity_manager,
                                                    **sleep_model_cfg)

            self.activity_manager.register_activity_controller(sleep_module)

        return sleep_module

    def create_local_activity_module(self):
        location_controller = LocationActivitiesController(self.population)
        self.activity_manager.register_activity_controller(location_controller)
        return location_controller

    def create_activity_manager(self):
        write_diary = self.config.get("activity_manager", {}).get("write_diary", False)
        return ActivityManager(self.population, self.relocator, write_diary)

    def create_default_activity_controller(self):
        default_activity = DefaultActivityController(self.population)
        self.activity_manager.register_activity_controller(default_activity, z_order=4)
        return default_activity

    def create_night_out_module(self, night_out, scenario):
        population = self.population
        night_out_module = None
        if night_out:
            night_out_module_cfg = self.config.get("night_out_module", {})
            night_out_module = NightOut(population, relocator=self.relocator, venues=scenario.venues(),
                                        **night_out_module_cfg)

        return night_out_module

    def create_daily_schedule_module(self, use_buses, scenario):
        def select_bus_wrapper(bus_selection_function, buses):
            def bus_selector():
                return bus_selection_function(buses)

            return bus_selector

        population = self.population
        schedules_cfg = self.config.get("daily_schedule", {})
        office_schedule = schedules_cfg["full_time_office_schedule"]
        schedules = np.array([Schedule(office_schedule) for _ in range(len(population))], ndmin=2).T
        part_time_percentage = schedules_cfg["part_time_workers"]
        part_time_schedule = schedules_cfg["part_time_office_schedule"]
        num_part_time_workers = int(len(self.population) * part_time_percentage)
        all_part_time_idx = []
        for home in sorted(scenario.homes, key=lambda x: len(x.inhabitants), reverse=True):
            num_to_choose = int(len(home.inhabitants) * .42) + 1

            if num_to_choose > num_part_time_workers:
                num_to_choose = num_part_time_workers

            part_time_idx = np.random.choice(home.inhabitants.index, num_to_choose, replace=False)
            schedules[part_time_idx] = np.array([Schedule(part_time_schedule) for _ in range(len(part_time_idx))], ndmin=2).T
            num_part_time_workers -= len(part_time_idx)
            all_part_time_idx.extend(part_time_idx)
            if num_part_time_workers <= 0:
                break

        if use_buses:
            pt_commuters = schedules_cfg["pt_commuters"]
            commuters_schedules = schedules_cfg["commuters_schedules"]
            commuter_groups = commuters_schedules.pop("groups", [])
            commuters_schedules["event_location"] = select_bus_wrapper(commuters_schedules["event_location"],
                                                                       scenario.buses)
            assigned_commuters = 0
            for group in commuter_groups:
                commuters_to_assign = int((len(pt_commuters) * group["proportion"]))
                if len(population) - (commuters_to_assign + assigned_commuters) < 0:
                    raise RuntimeError("Configuration error: Group proportions must add up to 1.")

                for ix in pt_commuters[assigned_commuters: assigned_commuters+commuters_to_assign]:
                    go_work = deepcopy(commuters_schedules)
                    go_home = deepcopy(commuters_schedules)
                    go_work["start_time"] = group["go_work"]
                    go_home["start_time"] = group["go_home"]
                    go_home["return_to"] = "home"
                    if ix in all_part_time_idx:
                        go_home["start_time"] = part_time_schedule[1]["start_time"]

                    schedules[ix, 0] = Schedule([go_work, go_home])

                assigned_commuters += commuters_to_assign

        schedule = ScheduleRoutines(population, relocator=self.relocator, schedule=schedules,
                                    must_follow_schedule=schedules_cfg.get("must_work", 1.),
                                    ignore_schedule=schedules_cfg.get("stays_home", None))
        return schedule

    def create_ambulance_module(self, hospitalize, scenario):
        population = self.population
        ambulance = None
        symptom_level = self.config.get("ambulance", {}).get("symptom_level", SymptomLevels.strong)
        if hospitalize:
            ambulance = Ambulance(population, self.relocator, scenario.hospital,
                                  symptom_level=symptom_level)

        return ambulance

    def create_intervention_modules(self, intervene, quarantine_households, test_to_exit):
        population = self.population
        ct_intervention = get_tested = tester = None
        if intervene:
            ct_intervention_cfg = self.config.get("ct_intervention", {})
            ct_intervention = ContactIsolationIntervention(population=population,
                                                           world=self.world,
                                                           relocator=self.relocator,
                                                           quarantine_household=quarantine_households,
                                                           **ct_intervention_cfg
                                                           )

            get_tested_cfg = self.config.get("get_tested", {})
            get_tested = GetTested(population,
                                   test_to_leave=test_to_exit,
                                   test_household=quarantine_households,
                                   **get_tested_cfg)

            tester_cfg = self.config.get("tester", {})
            tester = Test(population, **tester_cfg)

        return ct_intervention, get_tested, tester

    def create_contact_history_module(self, use_contact_history, scenario):
        if use_contact_history:
            population = self.population
            contact_history_cfg = self.config.get("contact_history", {})
            contact_history = ContactHistory(scenario.fnf_network, population=population, **contact_history_cfg)
            return contact_history

        return None

    def create_fnf_contact_tracing_module(self, intervene, scenario):
        population = self.population
        fnf_tracing = None
        if intervene:
            fnf_tracing_cfg = self.config.get("fnf_tracing", {})
            fnf_tracing = FriendsNFamilyContactTracing(scenario.fnf_network, population=population,
                                                       **fnf_tracing_cfg)

        return fnf_tracing

    def create_digital_contact_tracing_module(self, trace_contacts, dropout):
        population = self.population
        time_scalar = self.time_scalar
        c_tracing_cfg = self.config.get("dc_tracing", {})
        c_tracing = None
        if trace_contacts is True or trace_contacts in ["DCT", "both"]:
            c_tracing = RegionContactTracing(population=population,
                                             **c_tracing_cfg
                                             )

        return c_tracing

    def create_manual_contact_tracing_module(self, trace_contacts, scenario):
        population = self.population
        time_scalar = self.time_scalar
        mc_tracing_cfg = self.config.get("mc_tracing", {})
        m_tracing = None
        if trace_contacts in ["MCT", "both"]:
            m_tracing = ManualContactTracing(population=population,
                                             contact_network=scenario.social_network,
                                             **mc_tracing_cfg
                                             )

        return m_tracing

    def create_pathogen_model(self, use_pathogen):
        population = self.population
        covid = None
        if use_pathogen:
            config = self.config.get("pathogen", {}).copy()
            pathogen_class = config.pop("pathogen_class", RegionVirusDynamicExposure)
            covid = pathogen_class(population=population, **config)

        return covid

    def register_modules(self):
        modules_order = [
            # Contact tracing Modules
            self.contact_history,
            self.fnf_tracing,
            self.c_tracing,
            self.m_tracing,

            # Pathogen
            self.pathogen,

            # Intervention
            self.ct_intervention, self.get_tested, self.tester,
            self.undertaker,
            self.ambulance,

            # Behaviours
            self.daily_schedule,
            self.night_out,
            self.sleep_model,
            self.local_activity_model,
            self.activity_manager,
            self.motion
        ]
        for module in modules_order:
            if module is None:
                continue

            self.modules.append(module)

    def post_init_modules(self):
        # Post Init Regions
        for r in self.world.list_all_regions():
            r.post_init()

        self.engine.post_init_modules()

    def get_isolation_codes(self):
        return [self.get_tested is not None and self.get_tested.code or -1,
                self.fnf_tracing is not None and self.fnf_tracing.code or -1,
                self.m_tracing is not None and self.m_tracing.code or -1,
                self.c_tracing is not None and self.c_tracing.code or -1,
                self.ct_intervention is not None and self.ct_intervention.code or -1
                ]


class COVID19PartyContactTracing:
    def __init__(self, scenario, time_scalar, speed=0.7, radius=0.7):
        population = scenario.population
        self.world = scenario.world

        self.motion = MoveToTarget(self.world, population, speed=0.7)

        # Deceased duration distribution
        dd = partial(np.random.normal, 14 * time_scalar, 7 * time_scalar)
        id_ = partial(np.random.normal, 12 * time_scalar, 2 * time_scalar)

        # Deceased model
        self.covid = RegionCoronaVirusExposureWindow(radius=1.5, exposure_time=global_time.make_time(minutes=15),
                                                     population=population,
                                                     asymptomatic_p=0.4,
                                                     death_rate=0.05,
                                                     duration_distribution=dd,
                                                     incubation_distribution=id_,
                                                     susceptibility_window=global_time.make_time(hour=8)
                                                     )

        modules = []
        modules.extend(scenario.dynamic_regions())

        modules.extend([self.covid, self.motion])
        self.engine = Engine(modules)
        self.time_scalar = time_scalar
