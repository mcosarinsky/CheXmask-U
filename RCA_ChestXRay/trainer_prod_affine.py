import os
import torch
import torch.nn.functional as F
import argparse
import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset_affine import LandmarksDataset, ToTensor, RandomScaleRotate, AugColor
from siameseNet_affine import SiameseReg

def CrossVal(all_files, iFold, k = 5):
    #Performs 5-Fold-CrossValidation
    
    total = len(all_files)
    val = int(total/k)
    
    indices = list(range(total))
    
    train_indices = indices[0:(iFold-1)*val] + indices[iFold*val:]
    val_indices = indices[(iFold-1)*val:iFold*val]

    train_paths = [all_files[i] for i in train_indices]
    val_paths = [all_files[i] for i in val_indices]
    
    return train_paths, val_paths


def trainer(target_dataset, source_dataset, val_target_dataset, val_source_dataset, model, config):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
    val_target_loader = torch.utils.data.DataLoader(val_target_dataset, batch_size = config['val_batch_size'], num_workers = 0)
    val_source_loader = torch.utils.data.DataLoader(val_source_dataset, batch_size = config['val_batch_size'], num_workers = 0)

    # only train layers self.fc1 and self.fc_out
    #for param in model.encoder.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    train_rec_loss_avg = []
    val_loss_avg = []

    tensorboard = "Training"
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 1e12
    
    print('Training ...')
        
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])

    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        train_rec_loss_avg.append(0)
        num_batches = 0

        #if epoch == 100:
        #    print('Training encoder ...')
        #    for param in model.encoder.parameters():
        #        param.requires_grad = True

        iterator_target = iter(target_loader)
        iterator_source = iter(source_loader)

        for j in range(0, 200):
            sample_batched = next(iterator_target)
            x1, y1 = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            
            sample_batched = next(iterator_source)
            x2, y2 = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            
            params = model(x1, x2)
            
            affine_matrix = torch.zeros((params.shape[0], 3, 3), dtype=params.dtype, device=params.device)
            affine_matrix[:, 0, :] = params[:, 0:3]  
            affine_matrix[:, 1, :] = params[:, 3:6]  
            affine_matrix[:, 2, 2] = 1
                            
            y2 = torch.bmm(affine_matrix, y2.permute(0, 2, 1)).permute(0, 2, 1)
        
            recloss = F.mse_loss(y1[:, :, 0:2], y2[:, :, 0:2])
            
            if config['sampling'] == True:
                kld_loss_1 = -0.5 * torch.mean(torch.mean(1 + model.log_var_1 - model.mu_1 ** 2 - model.log_var_1.exp(), dim=1), dim=0)
                kld_loss_2 = -0.5 * torch.mean(torch.mean(1 + model.log_var_2 - model.mu_2 ** 2 - model.log_var_2.exp(), dim=1), dim=0)
                loss = recloss + 1e-5 * 0.5 * kld_loss_1 + 1e-5 * 0.5 * kld_loss_2
            else:
                loss = recloss
                
            optimizer.zero_grad()
            
            train_loss_avg[-1] += recloss.item()

            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            num_batches += 1
        
        train_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] train average reconstruction error: %f' % (epoch+1, config['epochs'], train_loss_avg[-1]*1024*1024))

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)

        iterator_source = iter(val_source_loader)
        iterator_target = iter(val_target_loader)

        with torch.no_grad():
            for j in range(0, 90):
                sample_batched = next(iterator_target)
                x1, y1 = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
                
                sample_batched = next(iterator_source)
                x2, y2 = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
                                
                params = model(x1, x2)
                
                affine_matrix = torch.zeros((params.shape[0], 3, 3), dtype=params.dtype, device=params.device)
                affine_matrix[:, 0, :] = params[:, 0:3]
                affine_matrix[:, 1, :] = params[:, 3:6]
                affine_matrix[:, 2, 2] = 1
                                
                y2 = torch.bmm(affine_matrix, y2.permute(0, 2, 1)).permute(0, 2, 1)
            
                recloss = F.mse_loss(y1[:, :, 0:2], y2[:, :, 0:2])
                
                val_loss_avg[-1] += recloss.item()
                num_batches += 1   

        val_loss_avg[-1] /= num_batches
        
        print('Epoch [%d / %d] validation average reconstruction error: %f' % (epoch+1, config['epochs'], val_loss_avg[-1] * 1024 * 1024))

        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)        
        writer.add_scalar('Validation/MSE', val_loss_avg[-1]  * 1024 * 1024, epoch)
                    
        if val_loss_avg[-1] < best:
            best = val_loss_avg[-1]
            print('Model Saved MSE')
            out = "bestMSE.pt"
            torch.save(model.state_dict(), os.path.join(folder, out))

        scheduler.step()
    
        torch.save(model.state_dict(), os.path.join(folder, "final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)    
    parser.add_argument("--model", default = "siameseReg", type=str)    
    parser.add_argument("--epochs", default = 1000, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 50, type = int)
    parser.add_argument("--gamma", default = 0.9, type = float)
    
    ## 5-fold Cross validation fold
    parser.add_argument("--fold", default = 1, type = int)
    
    config = parser.parse_args()
    config = vars(config)

    images_1 = open("train_images_lungs.txt",'r').read().splitlines()
    images_2 = open("test_images_lungs.txt",'r').read().splitlines()
    images = images_1 + images_2

    print(len(images))
    random.Random(13).shuffle(images)
        
    print('Fold %s'%config['fold'], 'of 10')
    images_train, images_val = CrossVal(images, config['fold'], 10)
    
    target_dataset = LandmarksDataset(images=images_train,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = False,
                                     transform = transforms.Compose([
                                                 RandomScaleRotate(5, 0.8),
                                                 AugColor(0.40),
                                                 ToTensor()])
                                     )

    source_dataset = LandmarksDataset(images=images_train,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = False,
                                     transform = transforms.Compose([
                                                 RandomScaleRotate(30, 0.7),
                                                 AugColor(0.40),
                                                 ToTensor()])
                                     )

    
    val_target_dataset = LandmarksDataset(images=images_val,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = False,
                                     transform = transforms.Compose([
                                                 RandomScaleRotate(5, 0.8),
                                                 AugColor(0.40),
                                                 ToTensor()])
                                     )

    val_source_dataset = LandmarksDataset(images=images_val,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = False,
                                     transform = transforms.Compose([
                                                 RandomScaleRotate(30, 0.7),
                                                 AugColor(0.40),
                                                 ToTensor()])
                                     )

    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    config['inputsize'] = 1024
    config['sampling'] = True
    
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SiameseReg(config).float()
    
    #load weights from previous training
    model.load_state_dict(torch.load("bestMSE.pt"), strict=False)
    #model.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/Training/pretrained_frozen/final.pt"), strict=False)
  
    
    trainer(target_dataset, source_dataset, val_target_dataset, val_source_dataset, model, config)