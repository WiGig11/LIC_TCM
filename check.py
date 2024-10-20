import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    # 获取可用 GPU 的数量
    num_gpus = torch.cuda.device_count()
    print(f"有 {num_gpus} 张可用的 GPU。")

    # 假设选择第一张显卡（索引为 0）
    device = torch.device("cuda:0")

    # 创建一个张量并放到指定显卡上
    tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print(f"张量在第一张显卡上：{tensor}")

    device = torch.device("cuda:1")

    # 创建一个张量并放到指定显卡上
    tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print(f"张量在第二张显卡上：{tensor}")
else:
    print("没有可用的 GPU，无法将张量放到显卡上。")