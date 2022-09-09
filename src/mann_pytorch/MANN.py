import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from mann_pytorch.GatingNetwork import GatingNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from mann_pytorch.MotionPredictionNetwork import MotionPredictionNetwork


class MANN(nn.Module):
    """Class for the Mode-Adaptive Neural Network."""

    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 num_experts: int, gn_hidden_size: int, mpn_hidden_size: int, dropout_probability: float):
        """Mode-Adaptive Neural Network constructor.

        Args:
            train_dataloader (DataLoader): Iterable over the training dataset
            test_dataloader (DataLoader): Iterable over the testing dataset
            num_experts (int): The number of expert weights constituting the Motion Prediction Network
            gn_hidden_size (int): The dimension of the 3 hidden layers of the Gating Network
            mpn_hidden_size (int): The dimension of the 3 hidden layers of the Motion Prediction Network
            dropout_probability (float): The probability of an element to be zeroed in the network training
        """

        # Superclass constructor
        super(MANN, self).__init__()

        # Retrieve input and output dimensions from the training dataset
        train_features, train_labels = next(iter(train_dataloader))
        input_size = train_features.size()[-1]
        output_size = train_labels.size()[-1]

        # Define the two subnetworks composing the MANN architecture
        self.gn = GatingNetwork(input_size=input_size,
                                output_size=num_experts,
                                hidden_size=gn_hidden_size,
                                dropout_probability=dropout_probability)
        self.mpn = MotionPredictionNetwork(num_experts=num_experts,
                                           input_size=input_size,
                                           output_size=output_size,
                                           hidden_size=mpn_hidden_size,
                                           dropout_probability=dropout_probability)

        # Store the dataloaders for training and testing
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mode-Adaptive Neural Network architecture.

        Args:
            x (torch.Tensor): The input vector for both the Gating and Motion Prediction networks

        Returns:
            y (torch.Tensor): The output of the Motion Prediction Network
        """

        # Retrieve the output of the Gating Network
        blending_coefficients = self.gn(x.T)

        # Retrieve the output of the Motion Prediction Network
        y = self.mpn(x, blending_coefficients=blending_coefficients)

        return y

    def train_loop(self, loss_fn: _Loss, optimizer: Optimizer, epoch: int, writer: SummaryWriter) -> None:
        """Run one epoch of training.

        Args:
            loss_fn (_Loss): The loss function used in the training process
            optimizer (Optimizer): The optimizer used in the training process
            epoch (int): The current training epoch
            writer (SummaryWriter): The updater of the event files to be consumed by TensorBoard
        """

        # Total number of batches
        total_batches = int(len(self.train_dataloader))

        # Cumulative loss
        cumulative_loss = 0

        # Print the learning rate and weight decay of the current epoch
        print('Current lr:', optimizer.param_groups[0]['lr'])
        print('Current wd:', optimizer.param_groups[0]['weight_decay'])

        # Iterate over batches
        for batch, (X, y) in enumerate(self.train_dataloader):

            # Compute prediction and loss
            pred = self(X.float()).double()
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update cumulative loss
            cumulative_loss += loss.item()

            # Periodically print the current average loss
            if batch % 1000 == 0:
                current_avg_loss = cumulative_loss/(batch+1)
                print(f"avg loss: {current_avg_loss:>7f}  [{batch:>5d}/{total_batches:>5d}]")

        # Print the average loss of the current epoch
        avg_loss = cumulative_loss/total_batches
        print("Final avg loss:", avg_loss)

        # Store the average loss, learning rate and weight decay of the current epoch
        writer.add_scalar('avg_loss', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('wd', optimizer.param_groups[0]['weight_decay'], epoch)
        writer.flush()

    def test_loop(self, loss_fn: _Loss) -> None:
        """Test the trained model on the test data.

        Args:
            loss_fn (_Loss): The loss function used for testing
        """

        # Dataset dimension
        num_batches = len(self.test_dataloader)

        # Cumulative loss
        cumulative_test_loss = 0

        with torch.no_grad():

            # Iterate over the testing dataset
            for X, y in self.test_dataloader:
                pred = self(X.float()).double()
                cumulative_test_loss += loss_fn(pred, y).item()

        # Print the average test loss at the current epoch
        avg_test_loss = cumulative_test_loss/num_batches
        print(f"Avg test loss: {avg_test_loss:>8f} \n")

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference step on the given input.

        Args:
            x (torch.Tensor): The input vector for both the Gating and Motion Prediction networks

        Returns:
            pred (torch.Tensor): The output of the Motion Prediction Network
        """

        with torch.no_grad():
            pred = self(x.float()).double()

        return pred

