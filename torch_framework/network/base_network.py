"""
Module containing the implementation of the base `Network` class
"""

import datetime
import sys
from time import time
from typing import Optional, Tuple

import numpy as np
import torch


class Network(torch.nn.Module):
    """
    Base `Network` class implementing training and evaluation loops
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Blank forward pass
        Child classes should implement this method
        """

    def evaluate(self,
                 test_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 loss_fn: torch.nn.Module,
                 return_preds: bool = False) -> Tuple[float, float, Optional[torch.Tensor]]:
        """
        Run inference on the provided data and report loss and accuracy

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            `DataLoader` object containing the test data

        device : torch.device
            `device` on which the evaluate is to be run

        loss_fn : torch.nn.Module
            Loss function used to compute the loss

        return_preds : bool, optional
            Flag to return the predictions

        Returns
        -------
        Tuple[float, float, Optional[torch.Tensor]]
            loss
                Average loss on the test set
            accuracy
                Accuracy on the test set (in range [0, 1])
        """

        self.eval()

        loss = 0
        num_correct = 0
        with torch.no_grad():
            if return_preds:
                predictions = []
            for data, target in test_loader:
                data = data.to(device)
                target = target.type(torch.LongTensor).to(device)

                output = self(data)
                loss += loss_fn(output, target).item() * len(data)

                preds = output.data.max(1, keepdim=True)[1]
                num_correct += preds.eq(target.data.view_as(preds)).sum()

                if return_preds:
                    predictions.append(preds.cpu().detach())

            if return_preds:
                predictions = torch.cat(predictions, dim=0)

        loss /= len(test_loader.dataset)
        accuracy = num_correct / len(test_loader.dataset)

        if return_preds:
            return loss, accuracy, predictions
        return loss, accuracy

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            epochs: int,
            device: torch.device,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            lr_schedule: torch.optim.lr_scheduler._LRScheduler = None,
            save_best_model: bool = False,
            save_monitor: str = "loss",
            save_filepath: str = None) -> dict:
        """
        Train the network on the given train data

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            `DataLoader` object containing the training data

        val_loader : torch.utils.data.DataLoader
            `DataLoader` object containing the validation data

        epochs : int
            Number of epochs to train on

        device : torch.device
            `device` on which the training is to be run

        optimizer : torch.optim.Optimizer
            Optimizer to update the parameters

        loss_fn : torch.nn.Module
            Loss function used to compute the loss

        lr_schedule : torch.optim.lr_scheduler._LRScheduler, optional
            `LRScheduler` object to change the learning rate over time
            Defaults to `None`.

        save_best_model : bool, optional
            Flag to save the best model
            Defaults to `False`.

        save_monitor : str, optional
            Metric to monitor for saving the best model
            Can be either `loss` or `acc`.
            Defaults to `loss`.

        save_filepath : str, optional
            File where the best model should be saved

        Returns
        -------
        dict
            Dictionary containing the loss and accuracy
            over the train and validation sets over the course of training
        """

        self.train()

        monitor = {}
        monitor_keys = ["train_acc", "train_loss", "val_acc", "val_loss"]
        for key in monitor_keys:
            monitor[key] = np.zeros(epochs)

        if save_best_model:
            if save_monitor == "loss":
                best_metric = float("inf")
            elif save_monitor == "acc":
                best_metric = float("-inf")
            else:
                raise ValueError(f"Unsupported `save_monitor`: {save_monitor}")
            if save_filepath is None:
                raise ValueError(f"No `save_filepath` provided.")

        for epoch in range(epochs):
            epoch_start_time = time()

            # Go over training data
            train_loss = 0
            train_correct = 0
            train_num = 0

            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                # Send images and labels to device
                data = data.to(device)
                target = target.type(torch.LongTensor).to(device)

                # Compute network output and update network weights
                optimizer.zero_grad(set_to_none=True)
                output = self(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                # Compute loss and accuracy
                train_loss += loss.item() * len(data)
                preds = output.data.max(1, keepdim=True)[
                    1]  # 1 is the dimension
                train_correct += preds.eq(target.data.view_as(preds)).sum()
                train_num += len(data)

                # Get running metrics
                current_loss = train_loss / train_num
                current_accuracy = train_correct / train_num

                # Compute time
                time_per_data = (time() - epoch_start_time) / train_num
                eta = (len(train_loader.dataset) - train_num) * time_per_data
                progress = train_num / len(train_loader.dataset)

                # Print stats
                print_string = f"Epoch {epoch+1}/{epochs}: "
                print_string += f"{batch_idx}/{len(train_loader)} "
                print_string += f"[{int(40*progress)*'='}>{int(40*(1-progress))*'.'}] - "
                print_string += f"ETA: {int(eta)}s - "
                if current_loss < 1e-3:
                    loss_str = f"{current_loss:.3e}"
                else:
                    loss_str = f"{current_loss:.3f}"
                print_string += f"loss: {loss_str} - acc: {current_accuracy:.4f}"
                sys.stdout.write(f"\r\033[K{print_string}")
                sys.stdout.flush()

            train_loss /= train_num
            train_accuracy = train_correct / train_num

            val_loss, val_accuracy = self.evaluate(val_loader, device, loss_fn)

            # Print stats
            print_string = f"Epoch {epoch+1}/{epochs}: "
            print_string += f"{len(train_loader)}/{len(train_loader)} "
            print_string += f"[{40*'='}] - "
            print_string += f"{datetime.timedelta(seconds=time() - epoch_start_time)} - "
            if train_loss < 1e-3:
                loss_str = f"{train_loss:.3e}"
            else:
                loss_str = f"{train_loss:.3f}"
            print_string += f"loss: {loss_str} - acc: {train_accuracy:.4f} - "
            if val_loss < 1e-3:
                loss_str = f"{val_loss:.3e}"
            else:
                loss_str = f"{val_loss:.3f}"
            print_string += f"val_loss: {loss_str} - val_acc: {val_accuracy:.4f}"
            sys.stdout.write(f"\r\033[K{print_string}\n")
            sys.stdout.flush()

            monitor["train_acc"][epoch] = train_accuracy
            monitor["train_loss"][epoch] = train_loss
            monitor["val_acc"][epoch] = val_accuracy
            monitor["val_loss"][epoch] = val_loss

            # Save best model
            if save_best_model:
                if save_monitor == "loss":
                    if val_loss < best_metric:
                        if best_metric < 1e-3:
                            metric_str = f"{best_metric:.3e}"
                        else:
                            metric_str = f"{best_metric:.3f}"
                        if val_loss < 1e-3:
                            val_str = f"{val_loss:.3e}"
                        else:
                            val_str = f"{val_loss:.3f}"

                        print(f"val_loss improved from {metric_str} to {val_str}")

                        torch.save(self.state_dict(), save_filepath)
                        best_metric = val_loss

                else:
                    if val_accuracy > best_metric:
                        print(f"val_acc improved from {best_metric:.4f} to {val_accuracy:.4f}")

                        torch.save(self.state_dict(), save_filepath)
                        best_metric = val_accuracy

                print(f"Best val_{save_monitor}: {best_metric}")

            if lr_schedule is not None:
                lr_schedule.step()

        return monitor
