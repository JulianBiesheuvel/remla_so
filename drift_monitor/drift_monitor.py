"""
Drift monitor.
"""

from time import time
from typing import List

from skmultiflow.drift_detection.adwin import ADWIN

prom_metrics = {"current_tick": 0, "num_errors": 0}


class DriftMonitor:
    """Class for monitoring drift over time."""

    def __init__(
        self,
        calculate_scores,
        metrics=List[str],
        detector=ADWIN,
    ) -> None:
        self.metrics = metrics
        self.last_scores = {}
        if isinstance(metrics, dict):
            self.metrics = metrics.keys()
            self.last_scores = metrics
        self.detectors = {m: detector() for m in self.metrics}
        self.calculate_scores = calculate_scores

    def tick(self):
        """Detects prediction drift"""
        prom_metrics["current_tick"] += 1
        try:
            scores = self.calculate_scores()
            for metric in self.metrics:
                self.detectors[metric].add_element(scores[metric])
            self.last_scores = scores
        except:  # pylint: ignore
            prom_metrics["num_errors"] += 1

    def prometheus_metrics(self):
        """Prometheus integration"""
        ts = int(time() * 1000)  # unix timestamp in [ms]

        return """\
# HELP Number of iterations so far
# TYPE drift_monitor_current_tick counter
drift_monitor_current_tick {current_tick} {ts}

# HELP Number of errors so far
# TYPE drift_monitor_num_errors counter
drift_monitor_num_errors {num_errors} {ts}
""".format(
            **prom_metrics, ts=ts
        ) + "\n".join(
            """
# HELP Last batch {k} score
# TYPE drift_monitor_{k} gauge
drift_monitor_{k} {v} {ts}

# HELP {k} drift detected
# TYPE drift_monitor_drift_detected_{k} gauge
drift_monitor_drift_detected_{k} {drift} {ts}

# HELP 1 if {k} is in the drift warning zone
# TYPE drift_monitor_drift_warning_{k} gauge
drift_monitor_drift_warning_{k} {warn} {ts}""".format(
                ts=ts,
                k=k,
                v=v,
                drift=int(self.detectors[k].detected_change()),
                warn=int(self.detectors[k].detected_warning_zone()),
            )
            for k, v in self.last_scores.items()
        )
