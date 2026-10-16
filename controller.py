"""Traffic signal controller.

Normal mode : lanes get green in cyclic order 1 -> 2 -> 3 -> 4 -> 1 ...
              green time = min_green + seconds_per_vehicle * vehicle_count (capped at max_green)
Emergency   : the moment any lane reports an emergency vehicle, every other lane is
              forced RED instantly and that lane is GREEN until the emergency vehicle
              has gone (lane.emergency becomes False), then the cycle resumes with the
              lane after it.

This module has no heavy dependencies so the logic can be tested on its own.
"""
import threading
import time


class SignalController:
    def __init__(self, lanes, get_settings, log):
        self.lanes = lanes
        self.n = len(lanes)
        self.get_settings = get_settings
        self.log = log
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self.signals = ["RED"] * self.n
        self.mode = "NORMAL"          # NORMAL | EMERGENCY
        self.em_lane = None
        self.current = 0
        self.phase = "GREEN"          # GREEN | YELLOW
        self.phase_start = time.time()
        self.phase_end = time.time()
        self.green_duration = 0

    # ------------------------------------------------------------------ helpers
    def green_time(self, count):
        s = self.get_settings()
        t = s["min_green"] + s["seconds_per_vehicle"] * count
        return int(round(max(s["min_green"], min(s["max_green"], t))))

    def _set_only(self, lane, colour):
        self.signals = ["RED"] * self.n
        self.signals[lane] = colour
        for l, s in zip(self.lanes, self.signals):
            l.signal = s

    def _start_green(self, lane):
        now = time.time()
        count = self.lanes[lane].count
        dur = self.green_time(count)
        self.current, self.phase = lane, "GREEN"
        self.phase_start, self.phase_end, self.green_duration = now, now + dur, dur
        self._set_only(lane, "GREEN")
        self.log(lane, "signal", f"{self.lanes[lane].name}: GREEN for {dur}s ({count} vehicles)")

    def _start_yellow(self):
        now = time.time()
        y = self.get_settings()["yellow_time"]
        self.phase, self.phase_start, self.phase_end = "YELLOW", now, now + y
        self._set_only(self.current, "YELLOW")

    def _enter_emergency(self, lane):
        prev = self.em_lane
        self.mode, self.em_lane, self.current, self.phase = "EMERGENCY", lane, lane, "GREEN"
        self.phase_start, self.phase_end, self.green_duration = time.time(), float("inf"), 0
        self._set_only(lane, "GREEN")  # all others RED instantly
        if prev is None:
            self.log(lane, "emergency", f"EMERGENCY at {self.lanes[lane].name}: all other lanes RED, lane GREEN")
        else:
            self.log(lane, "emergency", f"EMERGENCY moved to {self.lanes[lane].name}")

    # --------------------------------------------------------------------- tick
    def tick(self):
        now = time.time()
        with self.lock:
            em_lanes = [i for i, l in enumerate(self.lanes) if l.emergency]
            if self.mode == "NORMAL":
                if em_lanes:
                    first = min(em_lanes, key=lambda i: self.lanes[i].emergency_since or now)
                    self._enter_emergency(first)
                    return
                if now >= self.phase_end:
                    if self.phase == "GREEN":
                        self._start_yellow()
                    else:
                        self._start_green((self.current + 1) % self.n)
            else:
                lane = self.em_lane
                if self.lanes[lane].emergency:
                    return
                others = [i for i in em_lanes if i != lane]
                if others:
                    first = min(others, key=lambda i: self.lanes[i].emergency_since or now)
                    self._enter_emergency(first)
                    return
                # emergency vehicle has passed -> resume cyclic order after this lane
                self.log(lane, "emergency", f"Emergency cleared at {self.lanes[lane].name}: resuming normal cycle")
                self.mode, self.em_lane, self.current = "NORMAL", None, lane
                self._start_yellow()

    # ----------------------------------------------------------------- lifecycle
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        with self.lock:
            self._start_green(0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="signal-controller")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # keep the controller alive
                print("controller error:", exc)
            time.sleep(0.1)

    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self.lock:
            remaining = None if self.phase_end == float("inf") else max(0, int(self.phase_end - time.time() + 0.99))
            return {
                "signals": list(self.signals),
                "mode": self.mode,
                "em_lane": self.em_lane,
                "current": self.current,
                "phase": self.phase,
                "remaining": remaining,
                "green_duration": self.green_duration,
            }
