import os

CFG = dict(
    BASE_SEEDS     = list(range(30)),
    BUDGET_SECONDS = 10,
    TUNING_SEEDS   = list(range(10)),
    TUNING_INST    = 'berlin52',
    COLORS         = {'GA': '#2196F3', 'ACO': '#FF9800', 'CBGA': '#4CAF50'},
    N_JOBS         = os.cpu_count(),
    OUT_DIR        = 'outputs',
)

os.makedirs(CFG['OUT_DIR'], exist_ok=True)