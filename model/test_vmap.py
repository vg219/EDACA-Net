import torch

def dot(a, b, c):
    c2 = c.clone()  # 保持 c 的维度不变 [1, 4096, 2]
    a = a.view(-1, 1, 1)  # [4, 1, 1]
    b = b.view(-1, 1, 1)  # [4, 1, 1]
    # 扩展 c2 以匹配 vmap 的批处理维度 [4, 1, 4096, 2]
    c2 = c2.expand(4, *c2.shape[1:])  # [4, 1, 4096, 2]
    c2[..., 0] += a  # [4, 1, 4096] += [4, 1, 1]
    c2[..., 1] += b
    return c2

ax = torch.tensor([-1, 1, -1, 1], dtype=torch.float32)  # [4]
bx = torch.tensor([-1, -1, 1, 1], dtype=torch.float32)  # [4]
c = torch.randn(1, 4096, 2)  # [1, 4096, 2]

# 使用 vmap 遍历 ax 和 bx，但保持 c 的维度不变
yy = torch.vmap(dot, in_dims=(0, 0, None))(ax, bx, c)

# 由于每次 dot 返回的 c2 是 [1, 4096, 2]，vmap 会堆叠成 [4, 1, 4096, 2]
# 但我们只需要最后一次的结果，所以取最后一个元素
yy = yy[-1]  # [1, 4096, 2]

print(yy.shape)  # torch.Size([1, 4096, 2])