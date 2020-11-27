from typing import Any

import torch


class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.liner1 = torch.nn.Linear(D_in, H)
        self.liner2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.liner1(x).clamp(min=0)
        y_pred = self.liner2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(200):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
