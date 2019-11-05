import torch
import fire
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from utils import accuracy, xavier_normal_init, kl_loss
from models import BayesianNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(0)
np.random.seed(0)

def train(batch_size,
          epochs,
          model,
          dataset,
          valid_size=5000,
          label_smoothing=0.0,
          gpu='cuda:0'):

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    assert model in ['resnet18', 'resnet101', 'densenet121', 'densenet169']
    assert dataset in ['cifar10', 'cifar100']

    print("batch_size =", batch_size)
    print("epochs =", epochs)
    print("model =", model)
    print("data set =", dataset)
    print("label_smoothing =", label_smoothing)

    if dataset == 'cifar100':
        num_classes = 100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        train_set = datasets.CIFAR100('../data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                          transforms.RandomHorizontalFlip(),
                                          # transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(p=0.5),
                                          transforms.Normalize(mean=mean, std=std)
                                      ]))
        valid_set = datasets.CIFAR100('../data', train=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)]))
        train_indices = torch.load('./train_indices_cifar100.pth')
        valid_indices = torch.load('./valid_indices_cifar100.pth')
    elif dataset == 'cifar10':  # cifar10
        num_classes = 10
        mean = [0.4914, 0.48216, 0.44653]
        std = [0.2470, 0.2435, 0.26159]

        train_set = datasets.CIFAR10('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                         transforms.RandomHorizontalFlip(),
                                         # transforms.RandomRotation(15),
                                         transforms.ToTensor(),
                                         transforms.RandomErasing(p=0.5),
                                         transforms.Normalize(mean=mean, std=std)
                                     ]))
        valid_set = datasets.CIFAR10('../data', train=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)]))
        train_indices = torch.load('./train_indices_cifar10.pth')
        valid_indices = torch.load('./valid_indices_cifar10.pth')

    # indices = torch.randperm(len(train_set))
    # train_indices = indices[:len(indices) - valid_size]
    # valid_indices = indices[len(indices) - valid_size:]
    # torch.save(train_indices, './train_indices_' + dataset + '.pth')
    # torch.save(valid_indices, './valid_indices_' + dataset + '.pth')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))

    net = BayesianNet(num_classes=num_classes, model=model).to(device)
    net.apply(xavier_normal_init)

    # optimizer_net = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
    optimizer_net = optim.AdamW(net.parameters(), lr=0.01)
    lr_scheduler_net = optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, patience=10, factor=0.1)

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for e in range(epochs):
        net.train()

        epoch_train_loss = []
        epoch_train_acc = []
        is_best = False

        print("lr =", optimizer_net.param_groups[0]['lr'])
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer_net.zero_grad()
            logits = net(data)
            xent = F.cross_entropy(logits, target)
            kll = kl_loss(logits)
            loss = xent + label_smoothing*kll
            loss.backward()
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(accuracy(logits, target))
            optimizer_net.step()

        epoch_train_loss = np.mean(epoch_train_loss)
        epoch_train_acc = np.mean(epoch_train_acc)
        lr_scheduler_net.step(epoch_train_loss)

        net.eval()
        epoch_valid_loss = []
        epoch_valid_acc = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
                data, target = data.to(device), target.to(device)
                logits = net(data)
                loss = F.cross_entropy(logits, target)
                epoch_valid_loss.append(loss.item())
                epoch_valid_acc.append(accuracy(logits, target))

        epoch_valid_loss = np.mean(epoch_valid_loss)
        epoch_valid_acc = np.mean(epoch_valid_acc)

        print("Epoch {:d}: loss: {:4f}, acc: {:4f}, val_loss: {:4f}, val_acc: {:4f}"
              .format(e,
                      epoch_train_loss,
                      epoch_train_acc,
                      epoch_valid_loss,
                      epoch_valid_acc,
                      ))

        # save epoch losses
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        valid_losses.append(epoch_valid_loss)
        valid_accuracies.append(epoch_valid_acc)

        if valid_losses[-1] <= np.min(valid_losses):
            is_best = True

        if is_best:
            filename = f"../snapshots/{model}_best.pth.tar"
            print("Saving best weights so far with val_loss: {:4f}".format(valid_losses[-1]))
            torch.save({
                'epoch': e,
                'state_dict': net.state_dict(),
                'optimizer': optimizer_net.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accuracies,
                'val_losses': valid_losses,
                'val_accs': valid_accuracies,
            }, filename)

        if e == epochs-1:
            filename = f"../snapshots/{model}_{e}.pth.tar"
            print("Saving weights at epoch {:d}".format(e))
            torch.save({
                'epoch': e,
                'state_dict': net.state_dict(),
                'optimizer': optimizer_net.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accuracies,
                'val_losses': valid_losses,
                'val_accs': valid_accuracies,
            }, filename)


if __name__ == '__main__':
    fire.Fire(train)
