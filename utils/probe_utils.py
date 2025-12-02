from typing import Optional, Union
import torch
from tqdm import trange
from transformers import PreTrainedTokenizer, PreTrainedModel

class LinearProbe(torch.nn.Module):
    def __init__(
        self, input_size : int, output_size : int, 
        epochs : int = 100, device : Union[torch.device, str] = 'cuda', early_stopping : bool = True, patience : int = 10,
        loss_type : str = 'kl', learning_rate : float = 0.001
    ):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.epochs = epochs
        self.device = device
        self.early_stopping = early_stopping
        self.patience = patience
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.to(device)
    
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, X : torch.Tensor, y : torch.Tensor):
        """
        Fit the linear probe to the data.
        X: (n_samples, n_features)
        y: (n_samples, n_classes) or (n_samples)
        """
        X = X.to(self.device)
        y = y.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.loss_type == 'kl':
            criterion = torch.nn.KLDivLoss(reduction="batchmean")
        elif self.loss_type == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
            y = torch.argmax(y, dim=1)
        elif self.loss_type == 'mse':
            assert len(y.shape) == 1, f"Must be shape (n_samples) for regression, got {y.shape}"
            criterion = torch.nn.MSELoss()

        best_loss = float('inf')
        best_loss_epoch = 0
        with trange(self.epochs) as progress_bar:
            for epoch in progress_bar:
                optimizer.zero_grad()
                y_pred = self(X)
                if self.loss_type == 'kl':
                    y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
                if self.loss_type == 'mse':
                    y_pred = y_pred.squeeze(-1)
                loss = criterion(y_pred, y)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_loss_epoch = epoch
                if self.early_stopping and epoch - best_loss_epoch > self.patience:
                    print(f'Early stopping at epoch {epoch} (best loss: {best_loss})')
                    break

                loss.backward()
                progress_bar.set_postfix(loss=loss.item())
                optimizer.step()
    
    @torch.no_grad
    def score(self, X, y, device='cuda'):
        self.to(device)
        X = X.to(device)
        y = y.to(device)
        y_pred = self(X)
        if self.loss_type == 'kl':
            criterion = torch.nn.KLDivLoss(reduction="batchmean")
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
        elif self.loss_type == 'ce':
            criterion = lambda y_p, y_t: (y_p.argmax(dim=1) == y_t).float().mean()
            y = torch.argmax(y, dim=1)
        elif self.loss_type == 'mse':
            criterion = torch.nn.MSELoss()
            y_pred = y_pred.squeeze(-1)
        return criterion(y_pred, y).item()
    
    @torch.no_grad
    def pred(self, X, device='cuda'):
        self.to(device)
        X = X.to(device)
        return self(X)
    
class LinearProbeCV:
    def __init__(self, n_split : int = 5, **probe_kwargs):
        self.n_split = n_split
        self.probe_kwargs = probe_kwargs
        self.best_probe = None
        self.best_score = None

    def fit(self, X : torch.Tensor, y : torch.Tensor):
        """
        Fit the linear probe to the data.
        X: (n_samples, n_features)
        y: (n_samples, n_classes)
        """
        self.random_generator = torch.manual_seed(42)
        for _ in range(self.n_split):
            # random split into train (mask) and test (not mask)
            mask = torch.randperm(X.shape[0], generator=self.random_generator) % self.n_split != 0
            probe = LinearProbe(**self.probe_kwargs)
            probe.fit(X[mask], y[mask])
            score = probe.score(X[~mask], y[~mask])
            if self.best_score is None or score < self.best_score:
                self.best_score = score
                self.best_probe = probe
        return self
    
    def fit_splits(self, splits : list[tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]):
        """
        Fit the linear probe to the data.
        splits: list of tuples (X_train, y_train, X_val, y_val)
        """
        self.random_generator = torch.manual_seed(42)
        for X_train, y_train, X_val, y_val in splits:
            probe = LinearProbe(**self.probe_kwargs)
            probe.fit(X_train, y_train)
            score = probe.score(X_val, y_val)
            if self.best_score is None or score < self.best_score:
                self.best_score = score
                self.best_probe = probe
        return self

    def score(self, X, y):
        assert self.best_probe is not None
        return self.best_probe.score(X, y)
    
    def pred(self, X):
        assert self.best_probe is not None
        return self.best_probe.pred(X)

    

def get_activations(model : PreTrainedModel, X : dict, layer : int, batch_size : Optional[int] = None) -> torch.Tensor:
    model.eval()

    if batch_size is not None:
        activations = []
        for b in range(0, len(X['input_ids']), batch_size):
            batch_inputs = {
                'input_ids': X['input_ids'][b:b + batch_size],
                'attention_mask': X['attention_mask'][b:b + batch_size]
            }
            with torch.no_grad():
                batch_outputs = model(**batch_inputs, output_hidden_states=True)
                batch_activations = batch_outputs.hidden_states[layer].float().cpu()
                activations.append(batch_activations)
        activations = torch.cat(activations, dim=0)
        return activations

    with torch.no_grad():
        outputs = model(**X, output_hidden_states=True)
    
    activations = outputs.hidden_states[layer].squeeze().float().cpu()

    return activations

def get_token_alignment(sequence : torch.LongTensor, base_tokenizer : PreTrainedTokenizer, probe_tokenizer :PreTrainedTokenizer) -> dict[int, int]:
    """
    Create a mapping from the number of tokens in the base tokenizer to the number of tokens in the probe tokenizer.
    Used to align probes over token's residual streams & labels over token's logits.
    """
    token_alignment = {}
    for i in range(sequence.shape[0]):
        base_subseq = base_tokenizer.decode(sequence[:i+1]) # up and including i
        retokenized = probe_tokenizer(base_subseq, return_tensors='pt').input_ids[:, 1:] # skip BOS token
        token_alignment[i] = retokenized.shape[1] - 1 # get index of last token (parallel to i)
    return token_alignment