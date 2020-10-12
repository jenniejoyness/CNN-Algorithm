import torch.nn as nn
from gcommand_loader import GCommandLoader
import torch
import First_NN

train_path = './train'
valid_path = './valid'
test_path = './test'
#train_path = './ML4_dataset/data/train'
#valid_path = './ML4_dataset/data/valid'
#test_path = './ML4_dataset/data/test'
num_epochs = 15
num_classes = 30
image_size = 101 * 161
learning_rate = 0.001
batchsize = 100
batch_size_train = 100
batch_size_valid = 100
batch_size_test = 100
numWorkers1 = 20
numWorkers2 = 20
numWorkers3 = 20
TRUE = True
NONE = None
ZERO = 0
ONE = 1
first_fc_layer = 5880
second_fc_layer = 1000
third_fc_layer = 100




def train_data(model, optimizer, criterion, train_loader):

    # activate training mode
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            # run through the network
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # run back propagation and activate Adam optimisation
            back_prop(loss, optimizer)
            track_the_accuracy(labels, outputs, acc_list, i, total_step, epoch, loss)


def test_data(model, test_loader):

    # turn off training mode
    model.eval()
    with torch.no_grad():
        correct = ZERO
        total = ZERO
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, ONE)
            total += labels.size(ZERO)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test data: {} %'.format((correct / total) * 100))


def back_prop(loss, optimizer):

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def track_the_accuracy(labels, outputs, acc_list, i, total_step, epoch, loss):

    total = labels.size(ZERO)
    _, predicted = torch.max(outputs.data, ONE)
    correct = (predicted == labels).sum().item()
    acc_list.append(correct / total)

    if (i + 1) % batchsize == 0:
        print('epoch [{}/{}], acc: {:.2f}%'
              .format(epoch + 1, num_epochs, (correct / total) * 100))


def write_the_y_test_file(model, test_loader, audio_names):

    name_list = []
    list_y_hat = []
    for name in audio_names:
        name_list.append(name[0].split('/')[len(name[0].split('/')) - 1])

    for i, (data, labels) in enumerate(test_loader):
        outputs = model(data)
        _, y_hat = torch.max(outputs.data, 1)
        list_y_hat.extend(y_hat.tolist())

    file = open("test_y", "w")
    for i in range(len(audio_names)):
        file.write(name_list[i] + ', ' + str(list_y_hat[i]) + '\n')
    file.close()


def run_CNN(train_loader, valid_loader, test_loader, test_set):

    model = First_NN.FirstNet()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    critErion = nn.CrossEntropyLoss()

    train_data(model, opt, critErion, train_loader)
    test_data(model, valid_loader)
    write_the_y_test_file(model, test_loader, test_set)

def load_from_dataLoader(set,batchSize,shuffle,numWorkers):
    return torch.utils.data.DataLoader(
        set, batch_size=batchSize, shuffle=shuffle,
        num_workers=numWorkers, pin_memory=TRUE, sampler=NONE)

if __name__ == "__main__":

    dataset = GCommandLoader(train_path)
    train_loader = load_from_dataLoader(dataset,batch_size_train,TRUE,numWorkers1)

    validation_set = GCommandLoader(valid_path)
    valid_loader = load_from_dataLoader(validation_set,batch_size_valid,NONE,numWorkers2)

    test_set = GCommandLoader(test_path)
    test_loader = load_from_dataLoader(test_set,batch_size_test,NONE,numWorkers3)

    run_CNN(train_loader, valid_loader, test_loader, test_set.spects)
