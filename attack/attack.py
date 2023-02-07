model = ResNet20()

# hyperparams
batch_size = 128

# data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', download=True, transform=train_transform)

trainloader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)

model.cuda()
loss_func = nn.CrossEntropyLoss().cuda()

idx = 2

img, label = trainloader.dataset[idx]

model.zero_grad()
target_loss, _, _ = loss_func(model(ground_truth), label)
input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]
full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
print(f'Full gradient norm is {full_norm:e}.')
