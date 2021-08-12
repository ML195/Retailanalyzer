# Global path definitions

from pathlib import Path

PATHS = {
    'RAW_DATA_DIR': Path(__file__).parents[1].resolve() / 'data' / 'raw',
    'PROCESSED_DATA_DIR': Path(__file__).parents[1].resolve() / 'data' / 'processed',
    'MODELS_DIR': Path(__file__).parents[1].resolve() / 'models',
    'REPORTS_DIR': Path(__file__).parents[1].resolve() / 'reports',
    'EVALUATIONS_DIR': Path(__file__).parents[1].resolve() / 'reports' / 'evaluations',
    'RESULTS_DIR': Path(__file__).parents[1].resolve() / 'reports' / 'results'
}


