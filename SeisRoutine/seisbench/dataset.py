from math import cos, radians


def get_uncertainty(error_obj):
    """
    Safely extract uncertainty from ObsPy error objects.

    Parameters
    ----------
    error_obj : QuantityError or None

    Returns
    -------
    float or None
    """
    return getattr(error_obj, "uncertainty", None)


def get_source_params(event):
    """
    Extract source parameters from an ObsPy Event.

    Parameters
    ----------
    event : obspy.core.event.Event

    Returns
    -------
    dict or None
    """

    origin = event.preferred_origin()
    if origin is None and event.origins:
        origin = event.origins[0]

    if origin is None:
        return {}

    lat = origin.latitude
    lon = origin.longitude

    lat_unc_deg = get_uncertainty(origin.latitude_errors)
    lon_unc_deg = get_uncertainty(origin.longitude_errors)

    lat_unc_km = (
        lat_unc_deg * 111
        if lat_unc_deg is not None
        else None
    )

    lon_unc_km = (
        lon_unc_deg * 111 * cos(radians(lat))
        if lon_unc_deg is not None and lat is not None
        else None
    )

    source_params = {
        "source_id": str(event.resource_id),
        "source_origin_time": origin.time.datetime,
        "source_origin_uncertainty_sec":
            get_uncertainty(origin.time_errors),

        "source_latitude_deg": lat,
        "source_latitude_uncertainty_km": lat_unc_km,

        "source_longitude_deg": lon,
        "source_longitude_uncertainty_km": lon_unc_km,

        "source_depth_km":
            origin.depth / 1000
            if origin.depth is not None
            else None,

        "source_depth_uncertainty_km":
            get_uncertainty(origin.depth_errors) / 1000
            if get_uncertainty(origin.depth_errors) is not None
            else None,
    }

    return source_params


def get_source_quality_params(event):
    """
    Extract source quality information from an ObsPy Event.

    Parameters
    ----------
    event : obspy.core.event.Event

    Returns
    -------
    dict
    """

    origin = event.preferred_origin()
    if origin is None and event.origins:
        origin = event.origins[0]

    if origin is None:
        source_quality_params = {}
    else:
        quality = origin.quality
        origin_unc = origin.origin_uncertainty

        source_quality_params = {
            "source_azimuthal_gap_deg":
                getattr(quality, "azimuthal_gap", None),

            "source_used_phase_count":
                getattr(quality, "used_phase_count", None),

            "source_used_station_count":
                getattr(quality, "used_station_count", None),

            "source_standard_error":
                getattr(quality, "standard_error", None),

            "source_horizontal_uncertainty_km":
                (
                    getattr(origin_unc, "horizontal_uncertainty", None)
                    / 1000
                )
                if getattr(origin_unc, "horizontal_uncertainty", None)
                is not None
                else None,
        }

    return source_quality_params


def get_magnitude_params(event):
    """
    Extract magnitude information from an ObsPy Event.

    Parameters
    ----------
    event : obspy.core.event.Event

    Returns
    -------
    dict
    """

    mag = event.preferred_magnitude()

    if mag is None and event.magnitudes:
        mag = event.magnitudes[0]

    if mag is None:
        magnitude_params = {}
    else:
        magnitude_params = {
            "source_magnitude": mag.mag,
            "source_magnitude_uncertainty":
                get_uncertainty(mag.mag_errors),

            "source_magnitude_type":
                mag.magnitude_type,

            "source_magnitude_author":
                (
                    mag.creation_info.agency_id
                    if mag.creation_info is not None
                    else None
                ),
        }

    return magnitude_params


def get_event_params(event):
    """
    Extract all source metadata from an ObsPy Event.

    Parameters
    ----------
    event : obspy.core.event.Event

    Returns
    -------
    dict
    """

    event_params = {
        **get_source_params(event),
        **get_source_quality_params(event),
        **get_magnitude_params(event),
    }

    return event_params
