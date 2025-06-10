def translate_meta(location_params, env):
    """
    Used to produce system prompt.
    """
    prompt = "You are the HVAC administrator responsible for managing a building of type {0} located in {1}, where the climate is {2}. The building has {3} rooms in total."
    return prompt.format(
        location_params.build_type,
        location_params.city,
        location_params.climate_zone,
        env.roomnum,
    )
