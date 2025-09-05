import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy


class Training:
    def __init__(self, train_dataloader, val_dataloader=None, noisy_gradient=False):

        self.noisy_gradient = noisy_gradient

        self.losses = dict()
        
        self.validation = (val_dataloader is not None)
        self.train_dataloader = train_dataloader
        self.len_train = len(train_dataloader.dataset)
        self.losses["train"] = []

        if self.validation:
            self.val_dataloader = val_dataloader
            self.len_val = len(val_dataloader.dataset)
            self.losses["val"] = []
        else:
            self.losses["val"] = None

        self.best_val_loss = float('inf')
        self.best_model = None
    
    def add_gradient_noise(self, model, noise_std=0.01):
        """
        Adds Gaussian noise to the gradients of the model parameters.
        Called just before optimizer.step().
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Get the gradient
                    grad = param.grad
                    # Generate noise of the same shape as the gradient
                    noise = torch.randn_like(grad) * noise_std
                    # Add the noise to the gradient
                    param.grad.add_(noise)

    def training_loop(self, model, optimizer, num_epochs, gradient_noise_std=0.01):

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            # beta = c_annealer.get_beta(epoch) * model.beta

            for batch_data in self.train_dataloader:
                optimizer.zero_grad()
                
                model_output = model(batch_data)
                loss = model.loss(model_output, batch_data)
                
                loss.backward() # compute gradients
                if self.noisy_gradient:
                    self.add_gradient_noise(model, noise_std=gradient_noise_std)

                optimizer.step()
                
                train_loss += loss.item()

            
            self.losses["train"].append(train_loss / self.len_train)
            print(f'Epoch {epoch+1}, Loss: {train_loss / self.len_train:.4f}')
            
            if self.validation:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data in self.val_dataloader:
                        
                        model_output = model(batch_data)
                        loss = model.loss(model_output, batch_data)

                        val_loss += loss.item()
                self.losses["val"].append(val_loss / self.len_val)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss / self.len_val:.4f}')
            
                # save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = copy.deepcopy(model)
