import pandas as pd

def process_data(file_path):
    if 'txt' in file_path:
        file = pd.read_csv(file_path, header=None, sep=r'\s+')
        file = file.loc[:, [1, 3] + list(range(6, 15))]
        file = file[:-1]
        file['new'] = 'R'
        columns = range(0, 12)
        file.columns = columns
        is_attack = 1
    elif 'csv' in file_path:
        file = pd.read_csv(file_path)
        columns = range(0, 12)
        file.columns = columns
        is_attack = 1
    else:
        print("file_path不规范")
        return
    data = clean_data(file, is_attack)
    data = data.values
    return data

def clean_data(data, attack):
    # data = data.dropna(axis=0) // 该操作将有用信息全部删除了
    data = concatenate_columns(data)
    result = data[[0, 1, 2, 'new_column']].copy()
    if attack:
        col2_plus_2 = data[2] + 3
        result[3] = [data.loc[i, col] if col in data.columns else None
                     for i, col in enumerate(col2_plus_2)]
    result = result.dropna(axis=0)
    return result


def concatenate_columns(data):
    # 获取最大列数
    max_cols = data[2].astype(int) + 2
    max_col_overall = max_cols.max()

    # 创建一个空的结果Series
    result = pd.Series('', index=data.index)

    # 对每个可能的列进行向量化操作
    for col in range(3, max_col_overall + 1):
        if col in data.columns:
            # 只在需要该列的行中添加
            mask = max_cols > col - 1
            result[mask] += data[col][mask].astype(str)

    data['new_column'] = result
    return data