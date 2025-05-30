import os
import time
import shutil
import torch
import torch.utils
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from enhance import enhance_multiple_snr
from evaluate import evaluate
from utils import bold, copy_state, pull_metric, swap_state, LogProgress
from criteria import CompositeLoss

class Solver(object):
    def __init__(
        self, 
        data, 
        model, 
        optim, 
        args, 
        logger, 
        rank=0,     
        world_size=1, 
        device=None
    ):
        # Dataloaders and samplers
        self.tr_loader = data['tr_loader']      # Training DataLoader
        self.va_loader = data['va_loader']      # Validation DataLoader
        self.tt_loader_list = data['tt_loader_list']      # Test DataLoader for checking result samples
        self.ev_loader_list = data['ev_loader_list']      # Evaluation DataLoader
        self.tr_sampler = data['tr_sampler']    # Distributed sampler for training
        
        self.model = model
        self.optim = optim
        self.logger = logger

        # Basic config
        self.device = device or torch.device(args.device)
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        self.epochs = args.epochs
        self.loss = args.loss
        self.clip_grad_norm = args.clip_grad_norm
        
        self.input_type = args.model.input_type
        self.loss = CompositeLoss(args.loss).to(self.device)

        self.eval_every = args.eval_every   # interval for evaluation
            
        # Checkpoint settings
        self.checkpoint = args.checkpoint
        
        self.continue_from = args.continue_from
        
        if self.checkpoint and self.rank == 0:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.logger.info("Checkpoint will be saved to %s", self.checkpoint_file.resolve())

        self.writer = None
        self.best_state = None
        self.history = []
        self.log_dir = args.log_dir
        self.samples_dir = args.samples_dir
        self.num_prints = args.num_prints
        self.args = args
        
        # Initialize or resume (checkpoint loading)
        self._reset()

    def _serialize(self):
        """ Save checkpoint and best model (only rank=0).
        
            - We save both a 'checkpoint.th' file and a 'best.th' file.
            - The 'checkpoint.th' contains model state, optimizer state, and history.
            - The 'best.th' contains the best performing model state so far.
        """
        if self.rank != 0:  # save only on rank 0
            return
        
        # Create a package dict
        package = {}
        package['model'] = copy_state(self.model.module.state_dict() if self.is_distributed else self.model.state_dict())
        package['optimizer'] = self.optim.state_dict()
        package['history'] = self.history
        package['args'] = self.args
        
        # Write to a temporary file first
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        os.rename(tmp_path, self.checkpoint_file)

        # Save the best model separately to 'best.th'
        best_path = Path("best.th")
        tmp_path = str(best_path) + ".tmp"
        torch.save(self.best_state, tmp_path)
        os.rename(tmp_path, best_path)

    def _reset(self):
        """Load checkpoint if 'continue_from' is specified, or create a fresh writer if not."""
        if self.continue_from is not None:
            if self.rank == 0:
                self.logger.info(f'Loading checkpoint model: {self.continue_from}')
                if not os.path.exists(self.continue_from):
                    raise FileNotFoundError(f"Checkpoint directory {self.continue_from} not found.")
                
                # Attempt to copy the 'tensorbd' directory (TensorBoard logs) if it exists
                src_tb_dir = os.path.join(self.continue_from, 'tensorbd')
                dst_tb_dir = self.log_dir
                
                if os.path.exists(src_tb_dir):
                    # If the previous tensorboard logs exist, we either copy them
                    # to the new log dir or skip if it already exists.
                    if not os.path.exists(dst_tb_dir):
                        shutil.copytree(src_tb_dir, dst_tb_dir)
                    else:
                        # If the new log dir already exists, just issue a warning and do not overwrite
                        self.logger.warning(f"TensorBoard log dir {dst_tb_dir} already exists. Skipping copy.")
                    # Initialize the SummaryWriter to continue logging in the (possibly copied) directory
                    self.writer = SummaryWriter(log_dir=dst_tb_dir)
            
            package = None  # Initialize package to None for non-rank 0 processes
            
            # Rank 0 loads the checkpoint file from disk
            if self.rank == 0:
                ckpt_path = os.path.join(self.continue_from, 'checkpoint.th')
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
                self.logger.info(f"Loading checkpoint from {ckpt_path}")
                package = torch.load(ckpt_path, map_location='cpu')

            if self.is_distributed:
                # Wait until rank 0 finishes loading the checkpoint
                dist.barrier()
                
                # Broadcast the loaded checkpoint object to all ranks
                obj_list = [package]
                dist.broadcast_object_list(obj_list, src=0)
                package = obj_list[0] 

                # Extract model and optimizer state
                model_state = package['model']
                optim_state = package['optimizer']
                
                # Load states into the DDP-wrapped model and optimizer
                self.model.module.load_state_dict(model_state)
                self.optim.load_state_dict(optim_state)
            else:
                # In non-distributed (single GPU or CPU) mode, just load directly
                model_state = package['model']
                optim_state = package['optimizer']
                self.model.load_state_dict(model_state)
                self.optim.load_state_dict(optim_state)
            
            # Now attempt to load the best checkpoint if it exists
            best_path = os.path.join(self.continue_from, 'best.th')
            if os.path.exists(best_path):
                self.logger.info(f"Loading best model from {best_path}")
                self.best_state = torch.load(best_path, 'cpu')
                
            else:
                # If best.th does not exist, create a fallback best_state from the current model
                self.best_state = {
                    'model': copy_state(
                        self.model.module.state_dict() if self.is_distributed 
                        else self.model.state_dict()
                    )
                }
            
            # If there's any historical training metrics in the checkpoint, restore them
            if 'history' in package:
                self.history = package['history']
        else:
            # If there's no checkpoint to resume from, just create a fresh SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):
        # If there's a history from the checkpoint, replay metrics
        if self.history and self.rank == 0:  
            self.logger.info("Replaying metrics from previous run")
            for epoch, metrics in enumerate(self.history):
                info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
                self.logger.info(f"Epoch {epoch + 1}: {info}")
        
        if self.rank == 0:
            self.logger.info(f"Training for {self.epochs} epochs")
        
        # Main epoch loop
        for epoch in range(len(self.history), self.epochs):
            # Switch to training mode
            self.model.train()
            
            # If using a distributed sampler, set its epoch to ensure different random ordering each epoch
            if self.tr_sampler is not None:
                self.tr_sampler.set_epoch(epoch)
            
            start = time.time()
            if self.rank == 0:
                self.logger.info('-' * 70)
                self.logger.info("Training...")
            
            # Run one epoch of training
            train_loss = self._run_one_epoch(epoch)
            
            if self.rank == 0:
                self.logger.info(
                    bold(f'Train Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))
            
            # Optionally run validation if va_loader is present
            self.model.eval()
            
            if self.rank == 0:
                self.logger.info('-' * 70)
                self.logger.info('Validation...')
            
            with torch.no_grad():
                valid_loss = self._run_one_epoch(epoch, valid=True)
            
            if self.rank == 0:
                self.logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                        f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))

            
            # If distributed, we can synchronize here so that next epoch starts together
            if self.is_distributed:
                dist.barrier()
                
            # rank=0 handles model saving and test set evaluation
            if self.rank == 0:
                best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
                metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
                
                # Update best_state if we got a new best validation loss
                if valid_loss == best_loss:
                    self.logger.info(bold('New best valid loss %.4f'), valid_loss)
                    self.best_state = {'model': copy_state(self.model.module.state_dict() if self.is_distributed else self.model.state_dict())}

                # Evaluate on ev_loader (test set) every eval_every epochs (or last epoch)
                if self.eval_every is not None:
                    if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1):
                        self.logger.info('-' * 70)
                        self.logger.info('Evaluating on the test set...')
                        
                        # Temporarily swap model weights with best_state for evaluation
                        with swap_state(self.model.module if self.is_distributed else self.model, self.best_state['model']):
                            ev_metric = evaluate(self.args, self.model, self.ev_loader_list, epoch, self.logger)

                            self.logger.info('Enhance and save samples...')
                            enhance_multiple_snr(self.args, self.model, self.tt_loader_list, self.logger, epoch, self.samples_dir)
                    
                        for snr, metric_item in ev_metric.items():
                            for k, v in metric_item.items():
                                self.writer.add_scalar(f"test/{snr}/{k}", v, epoch)
                
                # Append metrics to history and print summary
                self.history.append(metrics)
                info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
                self.logger.info('-' * 70)
                self.logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))
                
                # Checkpoint serialization
                if self.checkpoint:
                    self._serialize()
                    self.logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
                    

    def _run_one_epoch(self, epoch, valid=False):
        """Run a single epoch of training or validation.
        
        Args:
            epoch (int): The current epoch index.
            valid (bool): Whether this is a validation epoch.
        Returns:
            float: The average loss over the epoch.
        """
        total_loss = 0.0
        data_loader = self.tr_loader if not valid else self.va_loader
        
         # If the sampler has a set_epoch method, call it to shuffle data consistently across ranks
        if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)

        label = ["Train", "Valid"][valid]
        name = label + f" | Epoch {epoch + 1}"
        
        # For rank=0, use a LogProgress, else just iterate the data
        if self.rank == 0:
            logprog = LogProgress(self.logger, data_loader, updates=self.num_prints, name=name)
        else:
            logprog = data_loader
        
        for i, data in enumerate(logprog):
            # Unpack data; if valid, there's an extra item (file id), else just 4 items
            if valid:
                tm, noisy_am, clean_am, mask = data
            else:
                tm, noisy_am, clean_am = data
                mask = None
            
            if mask is not None:
                mask = mask.to(self.device)
            
            # Move inputs to the correct device
            tm = tm.to(self.device)
            noisy_am = noisy_am.to(self.device)
            clean_am = clean_am.to(self.device)                    

            if self.input_type == "am":
                clean_am_hat = self.model(noisy_am)
            elif self.input_type == "tm":
                clean_am_hat = self.model(tm)
            elif self.input_type == "am+tm":
                clean_am_hat = self.model(tm, noisy_am)
            else:
                raise ValueError("Invalid input type, check input_type in config file.")
            
            # Compute loss
            loss_all, loss_dict = self.loss(clean_am_hat, clean_am, mask)
            
            # For distributed training, we do all_reduce
            if self.is_distributed:
                dist.all_reduce(loss_all)
                loss_all = loss_all / self.world_size
            
            total_loss += loss_all.item()

            if not valid:
                # Training step
                self.optim.zero_grad()
                
                if self.rank == 0:
                    # Log current losses in the progress bar
                    for i, (key, value) in enumerate(loss_dict.items()):
                        if i == 0:
                            logprog.update(**{key.capitalize(): format(value, "4.5f")})
                        else:
                            logprog.append(**{key.capitalize(): format(value, "4.5f")})
                        self.writer.add_scalar(f"train/{key.capitalize()}", value, epoch * len(data_loader) + i)
                    self.writer.add_scalar("train/Loss", loss_all.item(), epoch * len(data_loader) + i)
                
                # Backpropagation
                loss_all.backward()
                
                # Gradient clipping
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                    
                # Optimizer step
                self.optim.step()
                
            else:
                # Validation step (rank=0 logs)
                if self.rank == 0:
                    for i, (key, value) in enumerate(loss_dict.items()):
                        if i == 0:
                            logprog.update(**{key.capitalize(): format(value, "4.5f")})
                        else:
                            logprog.append(**{key.capitalize(): format(value, "4.5f")})
                    self.writer.add_scalar("valid/Loss", loss_all.item(), epoch * len(data_loader) + i)
        
        # Return the average loss over the entire epoch
        return total_loss / len(data_loader)