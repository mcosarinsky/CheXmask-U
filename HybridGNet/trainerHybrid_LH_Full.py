import os
import torch
import torch.nn.functional as F
import argparse
import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import scipy.sparse as sp
import numpy as np

from utils.utils import scipy_to_torch_sparse, genMatrixesLungs, genMatrixesLungsHeart, CrossVal
from utils.dataset_for_train import LandmarksDataset, ToTensor, RandomScale, AugColor, Rotate
from sklearn.metrics import mean_squared_error
from utils.metrics import hd_landmarks

from models.modelUtils import Pool
from models.HybridGNet2IGSC import Hybrid, HybridNoSkip
#from models.PCA import PCA_Net
#from models.FC import FC

def infinite_shuffled_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def linear_anneal(epoch, max_epochs, start=1e-5, end=1e-1):
    if epoch >= max_epochs:
        return end
    progress = epoch / max_epochs
    return start + progress * (end - start)

def trainer(train_dataset_lungs, val_dataset_lungs, train_dataset_heart, val_dataset_heart, model, config, downsample_lungs):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_loader_lungs = torch.utils.data.DataLoader(train_dataset_lungs, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
    val_loader_lungs = torch.utils.data.DataLoader(val_dataset_lungs, batch_size = config['val_batch_size'], num_workers = 0)

    train_loader_heart = torch.utils.data.DataLoader(train_dataset_heart, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
    val_loader_heart = torch.utils.data.DataLoader(val_dataset_heart, batch_size = config['val_batch_size'], num_workers = 0)

    lungs_iter = infinite_shuffled_iterator(train_loader_lungs)
    heart_iter = infinite_shuffled_iterator(train_loader_heart)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    train_rec_loss_avg = []
    val_loss_avg = []
    val_hd_avg = []

    tensorboard = "Weights"
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 1e12
    
    print('Training ...')
        
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    pool = Pool()
    
    for epoch in tqdm(range(config['epochs']), desc="Training Epochs"):
        model.train()
        model.kld_weight = linear_anneal(epoch, 200, start=1e-5, end=1e-2)

        train_loss_avg.append(0)
        train_rec_loss_avg.append(0)
        num_batches = 0
        
        for j in range(0, 100):
            coin = np.random.uniform(0, 1)

            if coin > 0.5:
                SAMPLE = "LUNGS"
                sample_batched = next(lungs_iter)
            else:
                SAMPLE = "HEART"
                sample_batched = next(heart_iter)
            
            image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            out = model(image)
            optimizer.zero_grad()
            
            if type(out) is not tuple:
                # PCA and FC
                B = target.shape[0]
                outloss = F.mse_loss(out, target.reshape(B,-1))
                loss = outloss
                
            elif (len(out)) == 3:
                # HybridGNet 2 IGSC
                if SAMPLE == "HEART":
                    target_down = pool(target, model.downsample_matrices[0])
                else:
                    target_down = pool(target, downsample_lungs[0])
                
                ts = target.shape[1]
                ts_d= target_down.shape[1]

                out, pre1, pre2 = out
                outloss = F.mse_loss(out[:,:ts,:], target) 
                loss = outloss

                if pre1 is not None and pre2 is not None:
                    pre1loss = F.mse_loss(pre1[:,:ts_d,:], target_down)
                    pre2loss = F.mse_loss(pre2[:,:ts,:], target)
                    loss += pre1loss + pre2loss
                
                kld_loss = -0.5 * torch.mean(torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0)
                loss += model.kld_weight * kld_loss

            else:
                raise Exception('Error unpacking outputs')

            train_rec_loss_avg[-1] += outloss.item()
            train_loss_avg[-1] += loss.item()

            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            num_batches += 1

        train_loss_avg[-1] /= num_batches
        train_rec_loss_avg[-1] /= num_batches

        print(f"Train average reconstruction error: {train_rec_loss_avg[-1] * 1024 * 1024:.1f}")

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)
        val_hd_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader_lungs:
                image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)

                out = model(image)
                if len(out) > 1:
                    out = out[0]

                out = out.reshape(-1, 2)
                target = target.reshape(-1, 2)
                ts = target.shape[0]
                
                dist_RL, dist_LL = hd_landmarks(out, target, 1024, False)
                dist = (dist_RL + dist_LL) / 2
                    
                val_hd_avg[-1] += dist 

                loss_rec = mean_squared_error(out[:ts,:].cpu().numpy(), target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
                num_batches += 1   
                loss_rec = 0

            for sample_batched in val_loader_heart:
                image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)

                out = model(image)
                if len(out) > 1:
                    out = out[0]

                out = out.reshape(-1, 2)
                target = target.reshape(-1, 2)
                
                dist_RL, dist_LL, dist_H = hd_landmarks(out, target, 1024, True)
                dist = (dist_RL + dist_LL + dist_H) / 3
                    
                val_hd_avg[-1] += dist 

                loss_rec = mean_squared_error(out.cpu().numpy(), target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
                num_batches += 1   
                loss_rec = 0

        val_loss_avg[-1] /= num_batches
        val_hd_avg[-1] /= num_batches
        
        print(f"Validation average reconstruction error: {val_loss_avg[-1] * 1024 * 1024:.1f}")

        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Train/MSE', train_rec_loss_avg[-1] * 1024 * 1024, epoch)
        
        writer.add_scalar('Validation/MSE', val_loss_avg[-1]  * 1024 * 1024, epoch)
        writer.add_scalar('Validation/Hausdorff Distance', val_hd_avg[-1], epoch)
                    
        if val_loss_avg[-1] < best:
            best = val_loss_avg[-1]
            print('Model Saved MSE')
            out = "bestMSE.pt"
            torch.save(model.state_dict(), os.path.join(folder, out))

        scheduler.step()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)    
    parser.add_argument("--model", default = "HybridGNet", type=str)    
    parser.add_argument("--epochs", default = 1000, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 50, type = int)
    parser.add_argument("--gamma", default = 0.9, type = float)
    parser.add_argument('--skip_connections', action='store_true', help='Use skip connections')

    # Number of filters at low resolution for HybridGNet
    parser.add_argument("--f", default = 32, type=int)
                
    # Define the output: only lungs, or lungs and heart by default
    parser.add_argument('--lungs', dest='Lungs', action='store_true')
    parser.set_defaults(Lungs=False)
    
    config = parser.parse_args()
    config = vars(config)

    print('Organs: Lungs and Heart')
    A, AD, D, U = genMatrixesLungsHeart()
    
    images_lungs = open("train_images_lungs.txt",'r').read().splitlines()
    images_heart = open("train_images_heart.txt",'r').read().splitlines()

    rng = random.Random(13)  
    
    print("Lungs:", len(images_lungs))
    print("Heart:", len(images_heart))
    rng.shuffle(images_lungs)
    rng.shuffle(images_heart)
    split_frac = 0.9
    
    split_L = int(len(images_lungs) * split_frac)
    images_train_L = images_lungs[:split_L]
    images_val_L = images_lungs[split_L:]

    split_H = int(len(images_heart) * split_frac)
    images_train_H = images_heart[:split_H]
    images_val_H = images_heart[split_H:]
    
    train_dataset_lungs = LandmarksDataset(images=images_train_L,
                                     img_path="../X-Ray/Chest-xray-landmark-dataset/Images",
                                     label_path="../X-Ray/Chest-xray-landmark-dataset/landmarks",
                                     organ = 'L',
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensor()])
                                     )

    val_dataset_lungs = LandmarksDataset(images=images_val_L,
                                     img_path="../X-Ray/Chest-xray-landmark-dataset/Images",
                                     label_path="../X-Ray/Chest-xray-landmark-dataset/landmarks",
                                     organ = 'L',
                                     transform = ToTensor()
                                     )

    train_dataset_heart = LandmarksDataset(images=images_train_H,
                                     img_path="../X-Ray/Chest-xray-landmark-dataset/Images",
                                     label_path="../X-Ray/Chest-xray-landmark-dataset/landmarks",
                                     organ = 'LH',
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensor()])
                                     )

    val_dataset_heart = LandmarksDataset(images=images_val_H,
                                     img_path="../X-Ray/Chest-xray-landmark-dataset/Images",
                                     label_path="../X-Ray/Chest-xray-landmark-dataset/landmarks",
                                     organ = 'LH',
                                     transform = ToTensor()
                                     )
                                                                        
 
    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    config['inputsize'] = 1024
    config['eval_sampling'] = False
    
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if config['model'] == 'HybridGNet':
        print('Model: HybrigGNet')
        
        f = int(config['f'])
        print(f, 'filters')
        config['filters'] = [2, f, f, f, f//2, f//2, f//2]
    
        N1 = A.shape[0]
        N2 = AD.shape[0]

        A = sp.csc_matrix(A).tocoo()
        AD = sp.csc_matrix(AD).tocoo()
        D = sp.csc_matrix(D).tocoo()
        U = sp.csc_matrix(U).tocoo()

        D_ = [D.copy()]
        U_ = [U.copy()]

        config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
        A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
        A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to('cuda:0') for x in X] for X in (A_, D_, U_))
    
        if config['skip_connections']:
            print('Using skip connections')
            model = Hybrid(config, D_t, U_t, A_t)
        else:
            print('No skip connections')
            model = HybridNoSkip(config, D_t, U_t, A_t)
        
    elif config['model'] == 'PCA':
        print('Model: PCA')
        model = PCA_Net(config)
    elif config['model'] == 'FC':
        print('Model: FC')
        model = FC(config)
    else:
        raise Exception('No valid model, choose between HybridGNet PCA or FC')
    
    _, _, DL, _ = genMatrixesLungs()
    DL = sp.csc_matrix(DL).tocoo()
    DL_ = [DL.copy()]
    DL_t= [scipy_to_torch_sparse(x).to('cuda:0') for x in DL_]
    
    trainer(train_dataset_lungs, val_dataset_lungs, train_dataset_heart, val_dataset_heart, model, config, DL_t)