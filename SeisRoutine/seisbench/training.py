import torch
import logging


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    stage,
    epoch,
    val_loss,
    ):
    """Save a complete training checkpoint."""

    checkpoint = {
        "stage": stage,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(checkpoint, path)
    
    
def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    ):
    """Load a training checkpoint."""

    checkpoint = torch.load(path, map_location=model.device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint


class EarlyStopping:
    def __init__(
        self,
        patience=10,
        min_improvement_percent=10,
        ):
        """
        min_improvement_percent:
            Minimum relative decrease in validation loss (%)
        """
        
        self.patience = patience
        self.min_improvement_percent = min_improvement_percent
        self.counter = 0
        self.best_loss = None
        
    def check(
            self,
            loss,
        ):
        if self.best_loss is None:
            self.best_loss = loss
            self.counter = 0
            msg = (
                "EarlyStopping, "
                "initialized with first validation loss, "
                f"{self.counter}/{self.patience}."
            )
            logging.info(msg)
            return False

        improvement = (self.best_loss - loss) / self.best_loss * 100
        if improvement >= self.min_improvement_percent:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        msg = (
            "EarlyStopping, "
            f"{improvement = :>4.2f}%, "
            f"{self.counter}/{self.patience}."
        )
        logging.info(msg)
        stop = self.counter >= self.patience
        
        return stop


def loss_fn1(y_pred, y_true):
    # vector cross entropy loss

    log_probs = torch.log_softmax(y_pred)
    h = y_true * log_probs
    # Mean along sample dimension and sum along pick dimension
    h = h.mean(-1).sum(-1)
    # Mean over batch axis
    h = h.mean()
    return -h

def loss_fn2(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    # Mean along sample dimension and sum along pick dimension
    h = h.mean(-1).sum(-1)
    # Mean over batch axis
    h = h.mean()
    return -h

def train_loop(
        model,
        loss_function,
        dataloader,
        optimizer,
    ):

    lst_loss = []
    total_loss = 0
    total_samples = 0
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        X = batch["X"].to(model.device)
        y = batch["y"].to(model.device)

        pred = model(X)
        loss = loss_function(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss = loss.item()
        total_loss += loss * X.size(0)
        total_samples += X.size(0)
        lst_loss.append((batch_id, loss))

        if batch_id % 5 == 0:
            current = batch_id * batch["X"].shape[0]
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    avg_loss = total_loss / total_samples

    return lst_loss, avg_loss

def test_loop(
        model,
        dataloader,
        loss_fn,
    ):

    total_loss = 0
    total_samples = 0


    was_training = model.training
    model.eval() # close the model for evaluation

    try:
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                # print(index, batch)
                X = batch["X"].to(model.device)
                y = batch["y"].to(model.device)
                pred = model(X)

                loss = loss_fn(pred, y)
                loss = loss.item()
                total_loss += loss * X.size(0)
                total_samples += X.size(0)
    finally:
        # re-open model for training stage
        model.train(was_training)

    avg_loss = total_loss / total_samples

    logging.info(f"Test avg loss: {avg_loss:>8f} \n")

    return avg_loss


