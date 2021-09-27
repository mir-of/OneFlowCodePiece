import oneflow as flow
import oneflow.nn as nn
import oneflow.utils.vision.transforms as transforms

from oneflow.nn.parallel import DistributedDataParallel as ddp

BATCH_SIZE = 128
DEVICE = flow.device("cuda")
WORLD_SIZE = flow.env.get_world_size()
RANK = flow.env.get_rank()

# Load Dataset
mnist_train = flow.utils.vision.datasets.MNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)
mnist_test = flow.utils.vision.datasets.MNIST(
    root="data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)

# DistributedSampler
train_sampler = flow.utils.data.distributed.DistributedSampler(
    mnist_train,
    num_replicas=WORLD_SIZE,
    rank=RANK)


# Dataloader
train_iter = flow.utils.data.DataLoader(
    mnist_train, BATCH_SIZE, shuffle=False, sampler=train_sampler
)
test_iter = flow.utils.data.DataLoader(
    mnist_test, BATCH_SIZE, shuffle=False
)


# Build Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(DEVICE)
model = ddp(model)
print(model)


# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)


def train(iter, model, loss_fn, optimizer):
    size = len(iter.dataset)
    for batch, (x, y) in enumerate(iter):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * BATCH_SIZE
        if batch % 100 == 0 and RANK == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(iter, model, loss_fn):
    size = len(iter.dataset)
    num_batches = len(iter)
    model.eval()
    test_loss, correct = 0, 0
    with flow.no_grad():
        for x, y in iter:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            test_loss += loss_fn(pred, y)
            bool_value = (pred.argmax(1).to(dtype=flow.int64) == y)
            correct += float(bool_value.sum().numpy())
    test_loss /= num_batches
    print("test_loss", test_loss.item(), "num_batches ", num_batches)
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f}")


def main():
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_sampler.set_epoch(t)
        train(train_iter, model, loss_fn, optimizer)
    test(test_iter, model, loss_fn)
    print("Done!")


if __name__ == '__main__':
    main()
