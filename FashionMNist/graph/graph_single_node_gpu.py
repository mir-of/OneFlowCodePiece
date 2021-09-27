import oneflow as flow
import oneflow.nn as nn
import oneflow.utils.vision.transforms as transforms

BATCH_SIZE = 128
DEVICE = flow.device("cuda")

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

# Dataloader
train_loader = flow.utils.data.DataLoader(
    mnist_train, BATCH_SIZE, shuffle=True
)
test_loader = flow.utils.data.DataLoader(
    mnist_test, BATCH_SIZE, shuffle=False, drop_last=True
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
print(model)

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)


# graph
class TrainGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return loss, y_pred


class EvalGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred


train_graph = TrainGraph()
eval_graph = EvalGraph()


def train(train_loader):
    size = len(train_loader.dataset)
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Compute
        losses, outputs = train_graph(x, y)

        current = batch * BATCH_SIZE
        if batch % 100 == 0:
            print(f"loss: {losses.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    print(f"size:{size}, num_batches:{num_batches}")
    # model.eval()
    test_loss, correct = 0, 0
    with flow.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            losses, outputs = eval_graph(x, y)
            
            test_loss += losses

            bool_value = (outputs.argmax(1).to(dtype=flow.int64) == y)
            correct += float(bool_value.sum().numpy())
    
    test_loss /= num_batches
    print("test_loss", test_loss, "num_batches ", num_batches)
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f}")


def main():
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader)
        test(test_loader)

    print("Done!")


if __name__ == '__main__':
    main()
