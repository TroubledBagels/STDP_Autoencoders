import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate
import network_model.GenerateData as GD
import snntorch.functional

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        self.sur_grad = snn.surrogate.fast_sigmoid()
        self.fc1 = torch.nn.Linear(1, 5, bias=False)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.sur_grad)
        self.fc2 = torch.nn.Linear(5, 5, bias=False)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.sur_grad)
        self.fc3 = torch.nn.Linear(5, 1, bias=False)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.sur_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_out = []

        for step in range(x.shape[1]):
            cur_input = x[:, step]

            cur_input = self.fc1(cur_input)
            spk1, mem1 = self.lif1(cur_input, mem1)

            cur_input = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur_input, mem2)

            cur_input = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur_input, mem3)

            spk_out.append(spk3)

        return torch.stack(spk_out, dim=1)

def train_basic(net, train_ds, test_ds):
    loss_rec = []
    test_mse_rec = []
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(100):
        net.train()

        loss_count = 0

        for inputs, labels in train_ds:
            inputs, labels = torch.Tensor(inputs), torch.Tensor(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_count += loss.item()

        print(f"Epoch {epoch}, train loss {loss_count/len(train_ds)}")
        loss_rec.append(loss_count)

        net.eval()
        loss_count = 0
        running_mse = 0
        for inputs, labels in test_ds:
            inputs, labels = torch.Tensor(inputs), torch.Tensor(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss_count += loss.item()
            running_mse += loss.item()

        print(f"Epoch {epoch}, test loss {running_mse/len(test_ds)}")
        test_mse_rec.append(running_mse/len(test_ds))



if __name__ == '__main__':
    basic = BasicNet()
    tr_ds = GD.SequenceDataset(num_samples=100, width=1, delay=1, spike_threshold=0.5)
    te_ds = GD.SequenceDataset(num_samples=100, width=1, delay=1, spike_threshold=0.5)

    train_basic(basic, tr_ds, te_ds)

