from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class FGVCDataPytorch:
    def __init__(self, num_classes=10, data_dir='./data', train_split_percent=0, batch_size=64, dataset=None, samples_per_class=None):
        self.data_dir = data_dir
        self.train_split_percent = train_split_percent
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class


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
        train_indices, _ = self._get_balanced_indices(labels)

        self.train_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_indices))
        #self.val_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(valid_indices))

        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

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
            # Optionally, you could add the remaining indices to valid_indices if needed
            valid_indices.extend(class_indices[self.samples_per_class:])

        print(len(train_indices))

        return train_indices, valid_indices

    def get_loaders(self):
        return self.train_loader, self.test_loader

