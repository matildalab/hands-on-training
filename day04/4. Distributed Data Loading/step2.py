import poptorch
import torchvision

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='../../datasets', download=True,
                                           transform=torchvision.transforms.ToTensor())
    options = poptorch.Options()
    dataloader = poptorch.DataLoader(options, dataset, batch_size=32, drop_last=True)
    for (data, label) in dataloader:
        print(f'* dataset size: {len(dataset)}\n'
              f'* dataloader size: {len(dataloader)}\n'
              f'* batch size: {data.shape[0]}')
        break
