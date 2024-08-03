from contextlib import contextmanager

import pystk
from tqdm import tqdm


def _initialize_pystk():
    cfg = pystk.GraphicsConfig.ld()
    cfg.screen_width = 128
    cfg.screen_height = 96
    try:
        pystk.init(cfg)
    except ValueError as e:
        raise ValueError("Try re-running if using a notebook") from e


@contextmanager
def rollout(
    callback: callable = None,
    max_steps: int = 100,
    frame_skip: int = 1,
    warmup: int = 20,
    track_name: str = "lighthouse",
    step_size: float = 0.1,
    disable_progress: bool = False,
):
    race_cfg = pystk.RaceConfig(track=track_name, step_size=step_size)
    race_cfg.num_kart = 1

    race = pystk.Race(race_cfg)
    race.start()
    race.restart()

    try:
        track = pystk.Track()
        state = pystk.WorldState()
        action = pystk.Action()

        for _ in range(warmup):
            race.step()
            state.update()

        def rollout_loop(race):
            for _ in tqdm(range(max_steps), disable=disable_progress):
                state.update()
                track.update()

                payload = {
                    "state": state,
                    "track": track,
                    "render_data": race.render_data,
                }

                yield payload

                if callback is not None:
                    action_dict = callback(**payload)
                    if not isinstance(action_dict, dict):
                        raise ValueError(f"Expecting action to be dict, got {type(action_dict)}")
                    for k, v in action_dict.items():
                        if not hasattr(action, k):
                            raise ValueError(f"Action has no attribute {k}")
                        setattr(action, k, v)

                for _ in range(frame_skip):
                    race.step(action)

        yield rollout_loop(race)
    finally:
        race.stop()
        del race


# initialize only once
if not globals().get('_pystk_init'):
    _initialize_pystk()
    globals()['_pystk_init'] = True
