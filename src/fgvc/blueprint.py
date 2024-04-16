from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class FGVCDataPytorch:
    def __init__(self, data_dir='./data', train_split_percent=0, batch_size=64, dataset=None, samples_per_class=None, num_workers=2, pin_memory=True):
        self.data_dir = data_dir
        self.train_split_percent = train_split_percent
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        datasets_switch = {
          'Aircraft': datasets.FGVCAircraft,
          'Pets': datasets.OxfordIIITPet,
          'Food101': datasets.Food101,
          'StanfordCars': datasets.StanfordCars,
          'Flowers': datasets.Flowers102,
          }

        self.dataset_class = datasets_switch[dataset]
        self.datasetname = dataset

        self._prepare_datasets()


    def _prepare_datasets(self):

        if self.datasetname == 'Pets':
          full_train_dataset = self.dataset_class(self.data_dir, split='trainval', download=True, transform=self.transform)
        elif self.datasetname == 'StanfordCars':
          full_train_dataset = self.dataset_class(self.data_dir, split='train', download=True, transform=self.transform)
        else:
          full_train_dataset = self.dataset_class(self.data_dir, split='train', download=True, transform=self.transform)
        test_dataset = self.dataset_class(self.data_dir, split='test', download=True, transform=self.transform)

        labels = [label for _, label in full_train_dataset]
        train_indices, valid_indices = self._get_balanced_indices(labels)

        if self.datasetname in ['Pets', 'StanfordCars', 'Food101']:
          self.val_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(valid_indices), num_workers=self.num_workers, pin_memory=self.pin_memory)
        else:
           val_dataset = self.dataset_class(self.data_dir, split='val', download=True, transform=self.transform)
           self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

        self.train_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def _get_balanced_indices(self, labels):
        num_classes = len(set(labels))
        indices_per_class = {class_idx: [] for class_idx in range(num_classes)}

        for idx, label in enumerate(labels):
            indices_per_class[label].append(idx)

        train_indices = []
        valid_indices = []

        for class_indices in indices_per_class.values():
            np.random.shuffle(class_indices)
            train_indices.extend(class_indices[:self.samples_per_class])
            valid_indices.extend(class_indices[self.samples_per_class:])

        print(f'Train/Val indice length: {len(train_indices)}, {len(valid_indices)}')

        return train_indices, valid_indices

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

