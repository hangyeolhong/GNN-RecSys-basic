import torch
from GATLayer import GATConv
# from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid_dim = 8
        self.in_dim = 8
        self.out_dim = 1

        self.conv1 = GATConv(dataset.num_features, self.hid_dim, heads=self.in_dim, dropout=0.6)
        self.conv2 = GATConv(self.hid_dim * self.in_dim, dataset.num_classes, concat=True,
                             heads=self.out_dim, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


@torch.no_grad()
def test(data):
    model.eval()
    res = model(data)
    pred = res.argmax(dim=1)
    correct = pred.eq(data.y.to(device))

    train_acc = correct[data.train_mask].sum().item() / data.train_mask.shape[0]
    test_acc = correct[data.test_mask].sum().item() / data.test_mask.shape[0]

    return train_acc, test_acc


dataset_name = 'Cora'
dataset = Planetoid(root='../data/' + dataset_name, name=dataset_name)
dataset.transform = T.NormalizeFeatures()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = GAT().to(device)
data = dataset[0].to(device)
print(data.x.shape, data.edge_index.shape)  # 2708 * 8, 2 * 10556

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    if epoch % 200 == 0:
        train_acc, test_acc = test(data)
        print("loss: %.4f\ttrain acc: %.4f\ttest acc: %.4f" % (loss.item(), train_acc, test_acc))

    loss.backward()
    optimizer.step()
