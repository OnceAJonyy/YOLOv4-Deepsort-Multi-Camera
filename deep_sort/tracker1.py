# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter1
from . import linear_assignment1
from . import iou_matching1
from .track1 import Track1


class Tracker1:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : 1kalman_filter1.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks1 : List[Track]
        The list of active tracks1 at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=60, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter1.KalmanFilter()
        self.tracks1 = []
        self._next_id = 1

    def predict1(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track1 in self.tracks1:
            track1.predict1(self.kf)

    def update1(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks1[track_idx].update1(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks1[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks1 = [t for t in self.tracks1 if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id1 for t in self.tracks1 if t.is_confirmed1()]
        features, targets = [], []
        for track1 in self.tracks1:
            if not track1.is_confirmed1():
                continue
            features += track1.features
            targets += [track1.track_id1 for _ in track1.features]
            track1.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        

    def _match(self, detections):

        def gated_metric(tracks1, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature1 for i in detection_indices])
            targets = np.array([tracks1[i].track_id1 for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment1.gate_cost_matrix(
                self.kf, cost_matrix, tracks1, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks1.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks1) if t.is_confirmed1()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks1) if not t.is_confirmed1()]

        # Associate confirmed tracks1 using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment1.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks1, detections, confirmed_tracks)

        # Associate remaining tracks1 together with unconfirmed tracks1 using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks1[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks1[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment1.min_cost_matching(
                iou_matching1.iou_cost, self.max_iou_distance, self.tracks1,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name1 = detection.get_class()
        self.tracks1.append(Track1(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature1, class_name1))
        self._next_id += 1
