"""Train the model"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import pandas as pd
import utils
import model.net as net
from model.data_loader import PhysioNetDataset 
from evaluate import evaluate
from torch.utils.data import DataLoader, random_split


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'last'


def train(model, optimizer, loss_fn, train_loader, metrics, params):
    """Train the model on all batches in epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(train_loader, unit="batch") as t:
        for i, data in enumerate(t):
            # fetch the next training batch
            train_batch = data['sx'].float()
            labels_batch = data['label'].float()

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch.unsqueeze(1))

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # compute sigmoid of output batch
                output_batch = torch.round(torch.sigmoid(output_batch))

                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return pd.DataFrame(summ)


def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validation data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    df_train = pd.DataFrame()
    df_eval = pd.DataFrame()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        train_metrics = train(model, optimizer, loss_fn, train_loader, metrics, params)
        df_train = pd.concat([df_train, train_metrics], ignore_index=True)

        # Evaluate for one epoch on validation set
        val_metrics, val_pd = evaluate(model, loss_fn, val_loader, metrics, params)
        df_eval = pd.concat([df_eval, val_pd], ignore_index=True)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        #save the weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    #Write the training and validation loss and accuracy data to a .csv
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    df_train.to_csv(f"train_cnn_data_{date}.csv")
    df_eval.to_csv(f"val_cnn_data_{date}.csv")


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    train_set = PhysioNetDataset(args.data_dir)
    val_set = PhysioNetDataset('test')
    train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=True)

    # specify the train and val dataset sizes
    params.train_size = len(train_set)
    params.val_size = len(val_set)

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
