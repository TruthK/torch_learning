import torch

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, dtype=dtype, device=device)
y = torch.randn(N, D_out, dtype=dtype, device=device)

w1 = torch.randn(D_in, H, dtype=dtype, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=dtype, device=device, requires_grad=True)

learning_rate = 1e-6
for i in range(200):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(i, loss)
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
