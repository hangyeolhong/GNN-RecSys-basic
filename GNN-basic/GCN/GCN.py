import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, x_cen):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Linearly transform node feature matrix
        x_cen = self.lin(x_cen)
        
        # Generate node embeddings(messages)
        h = torch.matmul(x, self.weight)
        
        # Propagate messages
        aggr_out = self.propagate(edge_index, size=None, h=h, edge_weight=None)

        return aggr_out + x_cen

    def message(self, h_j):
        return h_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)

    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index, x0)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv2(x1, edge_index, x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)
        
        x = self.lin(x2)
        return x.log_softmax(dim=-1)


@torch.no_grad()
def test(x0, edge_index):
    model.eval()
    res = model(x0, edge_index)
    pred = res.argmax(dim=1)
    correct = pred.eq(data.y.to(device))

    train_acc = correct[data.train_mask].sum().item() / data.train_mask.shape[0]
    test_acc = correct[data.test_mask].sum().item() / data.test_mask.shape[0]

    return train_acc, test_acc
    
    
dataset_name = 'Cora'
dataset = Planetoid(root='../data/' + dataset_name, name=dataset_name)
dataset.transform = T.NormalizeFeatures()  # normalize node features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


data = dataset[0].to(device)
model = GCN(hidden_channels=256, in_channel=dataset.num_features, out_channel=dataset.num_classes).to(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    if epoch % 200 == 0:
        train_acc, test_acc = test(data.x, data.edge_index)
        print("loss: %.4f\ttrain acc: %.4f\ttest acc: %.4f" % (loss.item(), train_acc, test_acc))

    loss.backward()
    optimizer.step()
