import json
import os
from collections import defaultdict


def calculate_average_slots(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 用于存储不同难度下的 slot 数量列表
    difficulty_slots = defaultdict(list)
    total_tasks = 0
    total_slots = 0  # 新增：用于记录所有任务的 slot 总数

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                task = json.loads(line)
                difficulty = task.get("difficulty", "unknown")
                slots = task.get("oracle_slots", [])

                slot_count = len(slots)
                difficulty_slots[difficulty].append(slot_count)

                total_slots += slot_count  # 累加总 slot 数
                total_tasks += 1
            except json.JSONDecodeError:
                print("JSON 解析错误，跳过该行。")

    print(f"==== 数据集统计: {os.path.basename(file_path)} ====")
    print(f"总任务数: {total_tasks}")
    print("-" * 40)

    for diff in ["easy", "middle", "hard", "unknown"]:
        if diff in difficulty_slots:
            counts = difficulty_slots[diff]
            avg_slots = sum(counts) / len(counts)
            print(f"难度: {diff:<7} | 样本数: {len(counts):<4} | 平均 Slot 数量: {avg_slots:.2f}")

    # 新增：计算并输出总的平均 oracle_slots
    if total_tasks > 0:
        overall_avg = total_slots / total_tasks
        print("-" * 40)
        print(f"全局统计  | 样本数: {total_tasks:<4} | 总平均 Slot 数量: {overall_avg:.2f}")


if __name__ == "__main__":
    # 替换为你实际的数据集路径
    dataset_path = "/home/hhr/home/InteractWeb-Bench/data/all.jsonl"
    calculate_average_slots(dataset_path)