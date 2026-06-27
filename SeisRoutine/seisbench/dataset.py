from math import sqrt
import numpy as np
from seisbench.util import stream_to_array
import SeisRoutine.catalog as src
import re


def build_phase_mapper(
        columns,
        families={"P", "S"},
    ):
    """
    Map arrival columns to their phase family.

    Example:
        trace_Pg_arrival_sample  -> P
        trace_Pn_arrival_sample  -> P
        trace_Sg_arrival_sample  -> S
        trace_Sg 2_arrival_sample -> S
    """
    pattern = re.compile(r"^trace_(.+?)_arrival_sample$")

    mapper = {}

    for col in columns:
        match = pattern.match(col)
        if not match:
            continue

        phase = match.group(1).strip()
        family = phase[0].upper()

        if family in families:
            mapper[col] = family

    return mapper


def find_ps_pairs(
        metadata
    ):
    """
    Return a boolean mask indicating rows that contain
    both P and S arrival picks.
    """
    mapper = build_phase_mapper(metadata.columns)

    p_cols = [col for col, phase in mapper.items() if phase == "P"]
    s_cols = [col for col, phase in mapper.items() if phase == "S"]

    p_exists = metadata[p_cols].notna().any(axis=1) if p_cols else False
    s_exists = metadata[s_cols].notna().any(axis=1) if s_cols else False

    ps_pairs = p_exists & s_exists

    return ps_pairs


class MetadataBuilder:

    def __init__(
        self,
        stream,
        event=None,
        inventory=None,
        component_order="ZNE",
        trace_category="earthquake",
    ):
        self.stream = stream
        self.event = event
        self.inventory = inventory
        self.component_order = component_order
        self.trace_category = trace_category

        self.trace = stream[0]

        self.starttime, self.data, self.completeness = stream_to_array(
            stream=stream,
            component_order=component_order,
        )

        # Filter picks for this station once; reused across all build methods.
        self._station_picks = self._filter_picks_for_station()
        self.origin = self._get_origin()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _filter_picks_for_station(self):
        """
        Return only the picks that belong to the station and network of this
        trace. Without this filter, picks from other stations would
        incorrectly appear in the metadata.
        """
        if self.event is None:
            return []

        tr = self.trace

        return [
            pick for pick in self.event.picks
            if (
                pick.waveform_id is not None
                and pick.waveform_id.station_code == tr.stats.station
                # and pick.waveform_id.network_code == tr.stats.network
            )
        ]

    @staticmethod
    def _get_uncertainty(error_obj):
        return getattr(error_obj, "uncertainty", None)

    def _get_origin(self):
        if self.event is None:
            return None

        origin = self.event.preferred_origin()

        if origin is None and self.event.origins:
            origin = self.event.origins[0]

        return origin

    def _get_magnitude(self):
        if self.event is None:
            return None

        magnitude = self.event.preferred_magnitude()

        if magnitude is None and self.event.magnitudes:
            magnitude = self.event.magnitudes[0]

        return magnitude

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    def build_source_parameters(self):
        origin = self.origin

        if origin is None:
            return {}

        # Use getattr to avoid AttributeError when an ObsPy Origin object has
        # quality or origin_uncertainty set to None.
        quality = getattr(origin, "quality", None)
        origin_unc = getattr(origin, "origin_uncertainty", None)

        depth_unc = self._get_uncertainty(origin.depth_errors)

        params = {
            "source_id":
                str(self.event.resource_id)
                if self.event.resource_id
                else None,

            "source_origin_time":
                origin.time.datetime,

            "source_origin_uncertainty_sec":
                self._get_uncertainty(origin.time_errors),

            "source_latitude_deg":
                origin.latitude,

            "source_latitude_uncertainty_deg":
                self._get_uncertainty(origin.latitude_errors),

            "source_longitude_deg":
                origin.longitude,

            "source_longitude_uncertainty_deg":
                self._get_uncertainty(origin.longitude_errors),

            "source_depth_km":
                origin.depth / 1000
                if origin.depth is not None
                else None,

            # ObsPy stores depth in metres; convert to kilometres.
            "source_depth_uncertainty_km":
                depth_unc / 1000
                if depth_unc is not None
                else None,

            "source_error_sec":
                getattr(quality, "standard_error", None)
                if quality is not None
                else None,

            "source_gap_deg":
                getattr(quality, "azimuthal_gap", None)
                if quality is not None
                else None,

            # ObsPy stores OriginUncertainty.horizontal_uncertainty in km,
            # so no unit conversion is needed here.
            "source_horizontal_uncertainty_km":
                getattr(origin_unc, "horizontal_uncertainty", None)
                if origin_unc is not None
                else None,
        }

        magnitude = self._get_magnitude()

        if magnitude is not None:
            params.update({
                "source_magnitude":
                    magnitude.mag,

                "source_magnitude_type":
                    magnitude.magnitude_type,

                "source_magnitude_author":
                    magnitude.creation_info.agency_id
                    if magnitude.creation_info
                    else None,
                "source_magnitude_uncertainty":
                    self._get_uncertainty(magnitude.mag_errors),
            })

        return params

    # ------------------------------------------------------------------
    # Station
    # ------------------------------------------------------------------

    def build_station_parameters(self):
        tr = self.trace

        params = {
            "station_code":           tr.stats.station,
            "station_network_code":   tr.stats.network,
            "station_location_code":  tr.stats.location,
            "station_latitude_deg":   None,
            "station_longitude_deg":  None,
            "station_elevation_m":    None,
            "station_sensitivity_counts_spm": None,
        }

        if self.inventory is None:
            return params

        # Geographic coordinates of the station.
        try:
            inv_sta = self.inventory.select(
                network=tr.stats.network,
                station=tr.stats.station,
            )[0][0]

            params.update({
                "station_latitude_deg":  inv_sta.latitude,
                "station_longitude_deg": inv_sta.longitude,
                "station_elevation_m":   inv_sta.elevation,
            })
        except Exception:
            pass

        # Channel sensitivity — was never populated in the original code.
        try:
            inv_cha = self.inventory.select(
                network=tr.stats.network,
                station=tr.stats.station,
                location=tr.stats.location,
                channel=tr.stats.channel,
            )[0][0][0]

            sensitivity = (
                inv_cha.response.instrument_sensitivity
                if inv_cha.response
                else None
            )

            if sensitivity is not None:
                params["station_sensitivity_counts_spm"] = sensitivity.value

        except Exception:
            pass

        return params

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def build_trace_parameters(self):
        tr = self.trace
        sps = tr.stats.sampling_rate
        origin = self.origin

        # stream_to_array returns one completeness value per component;
        # reduce to a single scalar for the metadata field.
        completeness = self.completeness
        if hasattr(completeness, "__len__"):
            completeness = float(np.mean(completeness))
        else:
            completeness = float(completeness)

        params = {
            "trace_name":
                f"{tr.id}__{self.starttime.isoformat()}",

            "trace_start_time":
                self.starttime.datetime,

            "trace_sampling_rate_hz":
                tr.stats.sampling_rate,

            "trace_dt_s":
                tr.stats.delta,

            "trace_npts":
                self.data.shape[-1],

            "trace_channel":
                tr.stats.channel[:2],

            "trace_category":
                self.trace_category,

            "trace_completeness":
                completeness,
        }

        # Per-component waveform statistics.
        for component, values in zip(self.component_order, self.data):
            params.update({
                f"trace_{component}_median_counts":
                    float(np.nanmedian(values)),

                f"trace_{component}_mean_counts":
                    float(np.nanmean(values)),

                f"trace_{component}_rms_counts":
                    float(np.sqrt(np.nanmean(values ** 2))),

                f"trace_{component}_min_counts":
                    float(np.nanmin(values)),

                f"trace_{component}_max_counts":
                    float(np.nanmax(values)),

                f"trace_{component}_lower_quartile_counts":
                    float(np.nanpercentile(values, 25)),

                f"trace_{component}_upper_quartile_counts":
                    float(np.nanpercentile(values, 75)),

                f"trace_{component}_spikes":
                    None,
            })

        # Phase picks (this station only).

        selector = src.CatalogPickArrivalSelector(
            picks=self._station_picks,
            arrivals=origin.arrivals,
        )
        for pick in self._station_picks:
            hint = (pick.phase_hint or "")

            if (hint=="") or (hint[0] not in "PS"):
                continue

            sample = int(round((pick.time - self.starttime) * sps))

            params[f"trace_{hint}_arrival_sample"] = sample

            params[f"trace_{hint}_status"] = (
                str(pick.evaluation_mode)
                if pick.evaluation_mode
                else None
            )

            # Pick weight lives on the Arrival object, not on Pick itself.
            pick_weight = None
            if origin is not None:
                arrival = selector.get_arrival_of_pick(pick=pick)
                # arrival = src.select_arrival_related_to_the_pick(
                #     pick=pick,
                #     arrivals=origin.arrivals,
                # )
                if arrival is not None:
                    pick_weight = arrival.time_weight

            params[f"trace_{hint}_weight"] = pick_weight

            params[f"trace_{hint}_uncertainty_s"] = (
                self._get_uncertainty(pick.time_errors)
                if pick.time_errors is not None
                else None
            )

        # Polarity is a single trace-level field (not per-phase).
        # Use the first pick for this station that carries a polarity value.
        for pick in self._station_picks:
            if pick.polarity:
                params["trace_polarity"] = str(pick.polarity)
                break
        else:
            params["trace_polarity"] = None

        return params

    # ------------------------------------------------------------------
    # Path
    # ------------------------------------------------------------------

    def build_path_parameters(self):
        origin = self.origin

        if origin is None or self.event is None:
            return {}

        params = {}

        selector = src.CatalogPickArrivalSelector(
            picks=self._station_picks,
            arrivals=origin.arrivals,
        )
        for pick in self._station_picks:
            arrival = selector.get_arrival_of_pick(pick=pick)
            # arrival = src.select_arrival_related_to_the_pick(
            #     pick=pick,
            #     arrivals=origin.arrivals,
            # )

            if arrival is None:
                continue

            hint = (pick.phase_hint or "")

            if (hint=="") or (hint[0] not in "PS"):
                continue

            params[f"path_{hint}_travel_s"] = float(
                pick.time - origin.time
            )

            params[f"path_{hint}_residual_s"] = arrival.time_residual

            params[f"path_weight_phase_location_{hint}"] = arrival.time_weight

            # arrival.distance in ObsPy is in degrees; convert to kilometres.
            if arrival.distance is not None:
                params["path_ep_distance_km"] = arrival.distance * 111.195

            if arrival.azimuth is not None:
                params["path_azimuth_deg"] = arrival.azimuth
                params["path_back_azimuth_deg"] = (arrival.azimuth + 180) % 360

        if "path_ep_distance_km" in params and origin.depth is not None:
            epi_km = params["path_ep_distance_km"]
            depth_km = origin.depth / 1000
            params["path_hyp_distance_km"] = sqrt(epi_km ** 2 + depth_km ** 2)

        return params

    # ------------------------------------------------------------------
    # All
    # ------------------------------------------------------------------

    def build_metadata(self):
        return {
            **self.build_trace_parameters(),
            **self.build_source_parameters(),
            **self.build_station_parameters(),
            **self.build_path_parameters(),
        }
