import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer):
        super(FeedForwardNet, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_layer)
        self.layer2 = nn.Linear(hidden_layer, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return F.log_softmax(x, dim=0)


def train_model(model, inputs, targets, optimizer, loss_func, num_epochs):
    for epoch in range(num_epochs):
        for i in range(len(inputs)):
            optimizer.zero_grad()
            
            out = model(inputs[i]).reshape(1,3)
            loss = loss_func(out, targets[i])

            loss.backward()
            optimizer.step()

def eval_model(model, inputs, targets):
    correct = 0
    with torch.no_grad():
        for i in range(len(inputs)):
            out = model(inputs[i]).reshape(1,3)
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets[i].view_as(pred)).sum().item()

    print('Got {0} out of {1} correct (Accuracy: {2:.4f})'.format(correct, len(inputs), 
                                                                correct / len(inputs)))

