from .optimized import Controller as OptimizedController


CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
STEER_MIN = -2.0
STEER_MAX = 2.0


class Controller(OptimizedController):
  """
  Warmup-aware tuned controller.

  The simulator ignores controller actions until step 100, so carrying a
  hypothetical `prev_action` through warmup misaligns the rate limiter with
  the action that was actually applied. This variant keeps its dynamic state
  neutral until the first real control step, then runs the tuned feedback law.
  """
  def __init__(self):
    super().__init__(params={
      "ff": 0.425,
      "p": 0.13,
      "i": 0.11,
      "d": 0.05,
      "look": 0.42020546122905866,
      "max_rate": 0.345,
      "long_look": 0.10058819594517204,
      "integral_decay": 0.9916823152834355,
      "aw_center": 1.831176190815695,
      "aw_fast_decay": 0.7338683577288903,
    })
    self.step_count = 0
    self.control_start_call = (CONTROL_START_IDX - CONTEXT_LENGTH) + 1

  def _reset_dynamic_state(self):
    self.integral = 0.0
    self.prev_error = 0.0
    self.prev_action = 0.0
    self.prev_target = None
    self.prev_preview = None

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.step_count += 1
    if self.step_count < self.control_start_call:
      self._reset_dynamic_state()
      return 0.0
    if self.step_count == self.control_start_call:
      self._reset_dynamic_state()

    action = super().update(target_lataccel, current_lataccel, state, future_plan)
    self.prev_action = float(max(STEER_MIN, min(STEER_MAX, self.prev_action)))
    return float(max(STEER_MIN, min(STEER_MAX, action)))
