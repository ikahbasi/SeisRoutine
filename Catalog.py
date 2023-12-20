def select_PickOfArrival(arrival, picks):
    origin = ev.preferred_origin()
    find_pick = False
    for pick in picks:
        if pick.resource_id == arrival.pick_id:
            find_pick = True
            break
    if not find_pick:
        pick = False
    return pick
