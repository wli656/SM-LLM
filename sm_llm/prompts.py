PROMPT_TYPES = ('P0', 'P1', 'P2', 'P3')

TASK_DESCRIPTION = (
    '[Task]: forecast surface soil moisture for the next {out_len} steps '
    'given the previous {in_len} steps'
)

VARIABLE_DESCRIPTION = (
    '[Variable Description]: SMAP stands for Soil Moisture Active Passive, a NASA satellite '
    'mission. SMAP provides surface soil moisture observations and serves as a key input '
    'variable for soil moisture forecasting. The variables clay and sand denote the percentage '
    'of clay and sand in the soil, and bd represents bulk density. NDVI stands for Normalized '
    'Difference Vegetation Index, a satellite-derived measure of vegetation health and density. '
    "The variables 'LC_0' to 'LC_7' are one-hot encoded indicators describing land cover types: "
    'Shrub (LC_0), Grassland (LC_1), Barren (LC_2), Wetland (LC_3), Forest (LC_4), '
    'Developed (LC_5), Cropland (LC_6), and Pasture (LC_7).'
)


def serialize_static(names, values, decimals=4):
    return '; '.join(f'{name} {value:.{decimals}f}' for name, value in zip(names, values))


def build_prompt(prompt_type, names, values, in_len, out_len):
    if prompt_type == 'P0':
        return ''
    if prompt_type == 'P1':
        return TASK_DESCRIPTION.format(out_len=out_len, in_len=in_len)

    static = '[Static data]: ' + serialize_static(names, values)
    if prompt_type == 'P2':
        return static
    if prompt_type == 'P3':
        return f'{VARIABLE_DESCRIPTION} {static}'
    raise ValueError(f'Unknown prompt type {prompt_type!r}; choose from {PROMPT_TYPES}.')
