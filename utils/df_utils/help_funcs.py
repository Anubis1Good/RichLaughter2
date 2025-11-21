


def deep_copy_dfs(old_dfs):
    # Глубокое копирование всей структуры
    dfs = {}
    for tf in old_dfs:
        dfs[tf] = {}
        for s in old_dfs[tf]:
            dfs[tf][s] = old_dfs[tf][s].copy(deep=True)
    return dfs