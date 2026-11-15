TASKS = {
    '15_5': dict(in_len=15, out_len=5, file='New_data_15_5_200loc.csv'),
    '90_30': dict(in_len=90, out_len=30, file='New_data_90_30_200loc.csv'),
}

TS_VARIABLES = ('eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST', 'SMAP')

TARGET_PREFIX = 'VWC_5'

LC_CODES = {
    'Shrub': 0, 'Grassland': 1, 'Barren': 2, 'Wetland': 3,
    'Forest': 4, 'Developed': 5, 'Crop': 6, 'Pasture': 7,
}
LC_COLUMNS = tuple(f'LC_{i}' for i in range(len(LC_CODES)))

STATIC_NUMERIC = (
    'DoY_normalized', 'Latitude', 'Longitude',
    'elevation', 'slope', 'aspect_normalized',
    'clay', 'bd', 'sand',
    'SMAP',
    'ndvi', 'evi', 'ndwi', 'lswi', 'nsdsi', 'cr', 'dpsvim', 'pol', 'rvim', 'vvvh',
)

STATIC_FEATURES = LC_COLUMNS + STATIC_NUMERIC

PROMPT_EXCLUDE = ('DoY_normalized', 'Latitude', 'Longitude')

LLM_REGISTRY = {
    'BERT': 'google-bert/bert-base-uncased',
    'SMOLLM': 'HuggingFaceTB/SmolLM2-135M',
    'GPT2': 'openai-community/gpt2',
    'TINYLLAMA': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'LLAMA': 'huggyllama/llama-7b',
}

EXPERIMENTS = {
    'table2': dict(task='15_5', train_years=1, targets=[2017, 2018, 2019, 2020, 2021, 2022]),
    'table3': dict(task='15_5', train_years=3, targets=[2019, 2020, 2021, 2022]),
    'table4': dict(task='90_30', train_years=3, targets=[2019, 2020, 2021, 2022]),
}

REPORTED_LEADS = {
    '15_5': [1, 2, 3, 4, 5],
    '90_30': [1, 7, 14, 21, 30],
}
