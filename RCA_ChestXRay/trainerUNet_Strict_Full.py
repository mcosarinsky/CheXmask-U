import os
import torch
import torch.nn.functional as F
import argparse
import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset_full import LandmarksDataset, ToTensorSeg, RandomScale, AugColor, Rotate
from unet import UNet, DiceLoss
from torch.nn import CrossEntropyLoss

from medpy.metric.binary import dc
import time

def evalImageMetricsL(output, target):
    dcp = dc(output == 1, target == 1)
    return dcp

def evalImageMetricsLH(output, target):
    dcp = dc(output == 1, target == 1)
    dcc = dc(output == 2, target == 2)
    return dcp, dcc

def evalImageMetricsLHC(output, target):
    dcp = dc(output == 1, target == 1)
    dcc = dc(output == 2, target == 2)
    dccla = dc(output == 3, target == 3)
    return dcp, dcc, dccla

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

def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['val_batch_size'], num_workers = 0)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    val_loss_avg = []
    val_dicelungs_avg = []
    val_diceheart_avg = []
    val_dicecla_avg = []

    tensorboard = "Training"
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 0
    suffix = ".pt"
    
    print('Training ...')
    
    dice_loss = DiceLoss().to(device)
    ce_loss = CrossEntropyLoss().to(device)
    
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    
    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        num_batches = 0
        
        t = time.time()

        for sample_batched in train_loader:
            image, target = sample_batched['image'].to(device), sample_batched['seg'].to(device)
            
            out = model(image)

            # backpropagation
            optimizer.zero_grad()
            
            loss = dice_loss(out, target) + ce_loss(out, target)
            train_loss_avg[-1] += loss.item()

            loss.backward()
            optimizer.step()

            num_batches += 1
        
        t2 = time.time()
        
        print('Training epoch took %.3f seconds' %(t2-t))

        train_loss_avg[-1] /= num_batches
        num_batches = 0

        model.eval()
        val_loss_avg.append(0)
        val_dicelungs_avg.append(0)
        val_diceheart_avg.append(0)
        val_dicecla_avg.append(0)
        
        t = time.time()
        with torch.no_grad():
            for sample_batched in val_loader:                
                image, target = sample_batched['image'].to(device), sample_batched['seg'].cpu().numpy()

                out = model(image)
                seg = torch.argmax(out[0,:,:,:], axis = 0).cpu().numpy()
                
                if config['organs'] == 'L':
                    dcl = evalImageMetricsL(seg, target[0,:,:])
                    val_dicelungs_avg[-1] += dcl
                    val_loss_avg[-1] += dcl

                elif config['organs'] == 'LH':
                    dcl, dch = evalImageMetricsLH(seg, target[0,:,:])
                    val_dicelungs_avg[-1] += dcl
                    val_diceheart_avg[-1] += dch
                    val_loss_avg[-1] += (dcl + dch) / 2

                elif config['organs'] == 'LHC':                    
                    dcl, dch, dcc = evalImageMetricsLHC(seg, target[0,:,:])
                    val_dicelungs_avg[-1] += dcl
                    val_diceheart_avg[-1] += dch
                    val_dicecla_avg[-1] += dcc
                    val_loss_avg[-1] += (dcl + dch + dcc) / 3

                num_batches += 1   

        val_loss_avg[-1] /= num_batches
        val_dicelungs_avg[-1] /= num_batches
        val_diceheart_avg[-1] /= num_batches
        val_dicecla_avg[-1] /= num_batches
        
        t2 = time.time()
        
        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Validation/Dice', val_loss_avg[-1], epoch)
        
        print('Epoch [%d / %d] validation Dice: %.3f, took %.3f seconds' % (epoch+1, config['epochs'], val_loss_avg[-1], t2-t))
    
        writer.add_scalar('Validation/Dice Lungs', val_dicelungs_avg[-1], epoch)
        
        if config['organs'] == 'L':
            print('Dice Lungs %.3f' %val_dicelungs_avg[-1])
        elif config['organs'] == 'LH':
            print('Dice Lungs %.3f. Dice Heart %.3f' %(val_dicelungs_avg[-1],val_diceheart_avg[-1]))
            writer.add_scalar('Validation/Dice Heart', val_diceheart_avg[-1], epoch)
        elif config['organs'] == 'LHC':   
            print('Dice Lungs %.3f. Dice Heart %.3f. Dice Clavicles %.3f' %(val_dicelungs_avg[-1],val_diceheart_avg[-1],val_dicecla_avg[-1]))
            writer.add_scalar('Validation/Dice Heart', val_diceheart_avg[-1], epoch)
            writer.add_scalar('Validation/Dice Cla', val_dicecla_avg[-1], epoch)    
        
        if val_loss_avg[-1] > best:
            best = val_loss_avg[-1]
            print('Model Saved Dice')
            out = "bestDice.pt"
            torch.save(model.state_dict(), os.path.join(folder, out))
            
        if epoch % 1 == 0:
            print('Model Saved')
            out = "epoch" + str(epoch) + suffix
            torch.save(model.state_dict(), os.path.join(folder, out))

        scheduler.step()

        print('')
        
    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)    
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default = 1024, type=int)
    parser.add_argument("--epochs", default = 10, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 3000, type = int)
    parser.add_argument("--gamma", default = 0.1, type = float)
    
    ## 5-fold Cross validation fold
    parser.add_argument("--fold", default = 1, type = int)
    parser.add_argument('--organs', type=str, default = 'L')
    
    config = parser.parse_args()
    config = vars(config)

    inputSize = config['inputsize']

    if config['organs'] == 'L':
        images = open("train_images_lungs.txt",'r').read().splitlines()
    elif config['organs'] == 'LH':
        images = open("train_images_heart.txt",'r').read().splitlines()
    else:
        raise ValueError('Organ not recognized')
       
    print(len(images))
    random.Random(13).shuffle(images)
        
    print('Fold %s'%config['fold'], 'of 5')
    images_train, images_val = CrossVal(images, config['fold'])
    
    train_dataset = LandmarksDataset(images=images_train,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = config['organs'],
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensorSeg()])
                                     )

    val_dataset = LandmarksDataset(images=images_val,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = config['organs'],
                                     transform = ToTensorSeg()
                                     )

    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    
    n_classes = len(config['organs']) + 1

    model = UNet(n_classes = n_classes)    
    trainer(train_dataset, val_dataset, model, config)