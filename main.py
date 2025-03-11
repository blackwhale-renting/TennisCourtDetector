from dataset import courtDataset
import torch
import torch.nn as nn
from base_trainer import train
from base_validator import val
import os
from tracknet import BallTrackerNet
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='number of steps per one epoch')
    args = parser.parse_args()
    
    train_dataset = courtDataset('train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_dataset = courtDataset('val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    exps_path = './exps/{}'.format(args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0)

    val_best_accuracy = 0

    for epoch in range(args.num_epochs):
        log = {}
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log.update(Train_training_loss=train_loss)

        if (epoch > 0) & (epoch % args.val_intervals == 0):
            val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, epoch)
            print('val loss = {}'.format(val_loss))

            log.update(Val_loss=val_loss)
            log.update(Val_tp=tp)
            log.update(Val_fp=fp)
            log.update(Val_fn=fn)
            log.update(Val_tn=tn)
            log.update(Val_precision=precision)
            log.update(Val_accuracy=accuracy)

            if accuracy > val_best_accuracy:
                print("New best accuracy = {}".format(accuracy), "Saving model...")
                val_best_accuracy = accuracy
                torch.save(model.state_dict(), model_best_path)     
                print('Model saved at {}'.format(model_best_path))

            
            torch.save(model.state_dict(), model_last_path)

        print('Epoch {}; Best accuracy = {}'.format(epoch, val_best_accuracy))
        print('Epoch', epoch, log)    

