import logging
import random
from typing import NamedTuple

from flatland.envs.malfunction_generators import malfunction_from_params
# from flatland.envs.rail_env import RailEnv
from envs.flatland.utils.gym_env_wrappers import FlatlandRenderWrapper as RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

MalfunctionParameters = NamedTuple('MalfunctionParameters', [('malfunction_rate', float), ('min_duration', int), ('max_duration', int)])


def random_sparse_env_small(random_seed, max_width, max_height, observation_builder):
    random.seed(random_seed)
    size = random.randint(0, 5)
    width = 20 + size * 5
    height = 20 + size * 5
    nr_cities = 2 + size // 2 + random.randint(0, 2)
    nr_trains = min(nr_cities * 5, 5 + random.randint(0, 5))  # , 10 + random.randint(0, 10))
    max_rails_between_cities = 2
    max_rails_in_cities = 3 + random.randint(0, size)
    malfunction_rate = 30 + random.randint(0, 100)
    malfunction_min_duration = 3 + random.randint(0, 7)
    malfunction_max_duration = 20 + random.randint(0, 80)

    rail_generator = sparse_rail_generator(max_num_cities=nr_cities, seed=random_seed, grid_mode=False,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rails_in_city=max_rails_in_cities)

    # new version:
    # stochastic_data = MalfunctionParameters(malfunction_rate, malfunction_min_duration, malfunction_max_duration)

    stochastic_data = {'malfunction_rate': malfunction_rate, 'min_duration': malfunction_min_duration,
                       'max_duration': malfunction_max_duration}

    schedule_generator = sparse_schedule_generator({1.: 0.25, 1. / 2.: 0.25, 1. / 3.: 0.25, 1. / 4.: 0.25})

    while width <= max_width and height <= max_height:
        try:
            env = RailEnv(width=width, height=height, rail_generator=rail_generator,
                          schedule_generator=schedule_generator, number_of_agents=nr_trains,
                          malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                          obs_builder_object=observation_builder, remove_agents_at_target=False)

            print("[{}] {}x{} {} cities {} trains, max {} rails between cities, max {} rails in cities. Malfunction rate {}, {} to {} steps.".format(
                random_seed, width, height, nr_cities, nr_trains, max_rails_between_cities,
                max_rails_in_cities, malfunction_rate, malfunction_min_duration, malfunction_max_duration
            ))

            return env
        except ValueError as e:
            logging.error(f"Error: {e}")
            width += 5
            height += 5
            logging.info("Try again with larger env: (w,h):", width, height)
    logging.error(f"Unable to generate env with seed={random_seed}, max_width={max_height}, max_height={max_height}")
    return None
