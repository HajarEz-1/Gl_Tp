from sklearn.ensemble import RandomForestRegressor


def build_model(random_state: int = 42,
                n_estimators: int = 200,
                max_depth: int | None = None,
                min_samples_split: int = 2,
                min_samples_leaf: int = 1,
                n_jobs: int | None = None):
    return RandomForestRegressor(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
    )

