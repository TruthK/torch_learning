def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


import torch
import numpy as np

NUM_DIGITS = 10
device = torch.device("cuda")


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

H = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 4)
)

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

BATCH_SIZE = 128

for epoch in range(1000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(epoch, loss)
    testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    testY = model(testX)
predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))
print('-----------------------------')
print([fizz_buzz_decode(i, x) for (i, x) in predictions])


print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])))
