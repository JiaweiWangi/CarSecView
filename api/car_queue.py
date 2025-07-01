from collections import deque, defaultdict

import numpy as np


def calculate_combined_stats(sizes, means, stds):
    # 验证输入的合法性
    if not (len(means) == len(stds) == len(sizes)):
        raise ValueError("均值、标准差和样本量的列表长度必须相同")

    # 计算总体均值
    total_size = sum(sizes)
    combined_mean = sum(m * n for m, n in zip(means, sizes)) / total_size

    # 计算总体标准差
    sum_sq_dev = sum(n * (std ** 2 + (m - combined_mean) ** 2) for m, std, n in zip(means, stds, sizes))
    combined_std = np.sqrt(sum_sq_dev / total_size)

    return combined_mean, combined_std


class StrideNode:
    def __init__(self, stride):
        self.stride = stride
        self.unique_id = set()
        self.temp_dlc = defaultdict(int)
        self.number = 0
        self.label = 1
        self.start_time = None

    def add_data(self, new_data):
        if self.start_time is None:
            self.start_time = new_data[0]
        if new_data[-1] == 'T':
            self.label = 0
        self.unique_id.add(new_data[1])
        self.temp_dlc[new_data[3]] += 1
        self.number += 1

    def get_result(self):
        return [self.label, self.unique_id, self.temp_dlc, self.number]

    def is_full(self, new_data):
        if self.number == 0:
            return False
        if new_data[0] - self.start_time > self.stride:
            return True
        else:
            return False


class CarQueue(deque):
    def __init__(self,max_len,stride,window_size):
        self.max_len = max_len
        self.stride = stride
        self.window_size= window_size
        self.node_len = int(self.window_size / self.stride)
        self.before = None
        super().__init__(maxlen=self.max_len)

    def get_result(self):
        index = 0
        results = []
        labels = []
        while index < self.max_len:
            number = 0
            label = 1
            unique_id = set()
            temp_dlc = defaultdict(int)
            for i in range(self.node_len):
                # print(temp)
                temp = self[index].get_result()
                for can in temp[1]:
                    unique_id.add(can)
                for k, v in temp[2].items():
                    temp_dlc[k] += v
                if temp[0] == 0:
                    label = 0
                index += 1
                number += temp[3]
            temp_dlc_1 = np.array([v for v in temp_dlc.values()])
            if number > 50:
                temp_res = [len(unique_id), np.mean(temp_dlc_1), np.std(temp_dlc_1)]
                self.before = temp_res
            else:
                temp_res = self.before
            results.append(temp_res)
            labels.append(label)

            # print(results)
        #print(len(results))
        return results, labels
