
import numpy as np


import torch.utils.data as Data




# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """


#    print (np.max(y), P.shape[0])

    # print(type(y[0][0]))

    # assert P.shape[0] == P.shape[1]
    # assert np.max(y) < P.shape[0]

    # row stochastic matrix
    # assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    # assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y




# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()

        

        # assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print (P)

    return y_train, actual_noise,P




def dataset_split_for_Animal(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10, noise_type='flip'):
    clean_train_labels = train_labels[:, np.newaxis]


    # if noise_type == 'flip':
    _, real_noise_rate, transition_matrix = noisify_pairflip(clean_train_labels,
                                                    noise=noise_rate, random_state=random_seed, nb_classes=num_classes)


    # if noise_type == 'symmetric':
    #     _, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels,
                                                # noise=noise_rate, random_state= random_seed, nb_classes=num_classes)
    
    # elif noise_type == 'asymmetric':
    #     _, real_noise_rate, transition_matrix = noisify_multiclass_asymmetric(clean_train_labels,
    #                                                 noise=noise_rate, random_state=random_seed, nb_classes=num_classes)

    # print(transition_matrix)

    noisy_labels = train_labels

    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels, transition_matrix
    





class Animal_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=1, random_seed=1, num_class=10, noise_type='symmetric', anchor=True):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.anchor = anchor
        # print(self.anchor)
        # input()
        if anchor:
            original_images = np.load('../../coteaching_plus_Animal/data/Animal_npy/train_images.npy')
            original_labels = np.load('../../coteaching_plus_Animal/data/Animal_npy/train_labels.npy')
        else:
            original_images = np.load('Animal_npy/mnist_images.npy')
            original_labels = np.load('Animal_npy/mnist_labels.npy')

        original_images=original_images.transpose((0,3,1,2))

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.t = dataset_split_for_Animal(original_images,
                                                                             original_labels, noise_rate, split_per, random_seed, num_class, noise_type)
        
        # print(self.train_data.shape)
        # print(self.val_data.shape)
        # input()
        
        print('we are training for Animal 10N')

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

     
        return img, label, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
   
        else:
            return len(self.val_data)
 


class Animal_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        self.test_data = np.load('../../coteaching_plus_Animal/data/Animal_npy/test_images.npy')
        self.test_labels = np.load('../../coteaching_plus_Animal/data/Animal_npy/test_labels.npy')  # 0-9
        
        self.test_data=self.test_data.transpose((0,3,1,2))


        print('we are testing for Animal 10N')

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label,index
    
    def __len__(self):
        return len(self.test_data)