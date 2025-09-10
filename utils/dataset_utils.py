from torch.utils.data import  Subset
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader



def get_classic_loaders(list_of_digits, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_indices = [i for i, label in enumerate(train.targets) if label in list_of_digits]
    test_indices = [i for i, label in enumerate(test.targets) if label in list_of_digits]

    train_subset = Subset(train, train_indices)
    test_subset = Subset(test, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
