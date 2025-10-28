import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))  # type: ignore


cache_path = 'aipipe_reprod/shapley/saves/cache'
cache_comb_path = 'aipipe_reprod/shapley/saves/cache_comb'
cache_hie_path = 'aipipe_reprod/shapley/saves/hierarchical_cache'

dataset_names = [
    'abalone',
    'ada_prior',
    'avila',
    'connect-4',
    'eeg',
    'google',
    'house_prices',
    'jungle_chess',
    'micro',
    'mozilla4',
    'obesity',
    'page-blocks',
    'pbcseq',
    'pol',
    'run_or_walk',
    'shuttle',
    'uscensus',
    'wall-robot-nav',
]

def merge_cache(dataset_name):
    with open(f'{cache_path}/{dataset_name}.json', 'r', encoding='utf-8') as f:
        cache: dict[str, float] = json.load(f)

    with open(f'{cache_path}/{dataset_name}_cache.json', 'r', encoding='utf-8') as f:
        cache2: dict[str, float] = json.load(f)

    with open(f'{cache_comb_path}/{dataset_name}.json', 'r', encoding='utf-8') as f:
        if os.path.getsize(f'{cache_comb_path}/{dataset_name}.json') == 0:
            cache_comb = {}
        else:
            cache_comb: dict[str, float] = json.load(f)

    with open(f'{cache_hie_path}/{dataset_name}_operator_cache.json', 'r', encoding='utf-8') as f:
        cache_hie: dict[str, float] = json.load(f)
        cache_hie = {key.replace('op:', ''): value for key, value in cache_hie.items()}

    with open(f'{cache_hie_path}/{dataset_name}_category_cache_mab.json', 'r', encoding='utf-8') as f:
        cache_hie2: dict[str, float] = json.load(f)
        cache_hie2 = {key.replace('op:', ''): value for key, value in cache_hie2.items()}
    
    cache.update(cache_comb)
    cache.update(cache2)
    cache.update(cache_hie)
    cache.update(cache_hie2)

    with open(f'{cache_path}/{dataset_name}_merged.json', 'w') as f:
        json.dump(cache, f)

    
    print(f'{dataset_name} merged')

def remove_prefix(dataset_name):
    with open(f'{cache_path}/{dataset_name}_merged.json', 'r', encoding='utf-8') as f:
        cache: dict[str, float] = json.load(f)
    cache = {key.replace('op:', '').replace('cat:', ''): value for key, value in cache.items()}
    with open(f'{cache_path}/{dataset_name}_merged.json', 'w') as f:
        json.dump(cache, f)

def remove_neg1(dataset_name):
    with open(f'{cache_path}/{dataset_name}_merged.json', 'r', encoding='utf-8') as f:
        cache: dict[str, float] = json.load(f)
    cache2 = [(key, value) for key, value in cache.items() if '-1' in key]
    pipe2 = [json.loads(key.split(':')[1]) for key, value in cache2]
    pipe2 = [list(filter(lambda x: x != -1, pipe)) for pipe in pipe2]
    cache2 = {f'{dataset_name}:{str(pipe)}': value for pipe, value in zip(pipe2, [value for _, value in cache2])}
    cache.update(cache2)
    # print(cache2)
    print(dataset_name, 'processed', len(cache2), 'records')
    with open(f'{cache_path}/{dataset_name}_merged.json', 'w') as f:
        json.dump(cache, f)


if __name__ == '__main__':
    for dataset_name in dataset_names:
        remove_neg1(dataset_name)
