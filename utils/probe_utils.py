from typing import Optional, Union
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import PreTrainedTokenizer, PreTrainedModel

class Probe(torch.nn.Module):
    def __init__(
        self,
        epochs : int = 100, 
        device : Union[torch.device, str] = 'cuda', 
        early_stopping : bool = True, 
        patience : int = 10,
        loss_type : str = 'kl', 
        learning_rate : float = 0.001
    ):
        super(Probe, self).__init__()
        self.epochs = epochs
        self.device = device
        self.early_stopping = early_stopping
        self.patience = patience
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.to(device)

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
        elif self.loss_type == "bce":
            assert (
                len(y.shape) == 1
            ), f"Must be shape (n_samples) for binary classification, got {y.shape}"
            criterion = torch.nn.BCEWithLogitsLoss()

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
                if self.loss_type == "bce":
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
        elif self.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
            y_pred = y_pred.squeeze(-1)
        return criterion(y_pred, y).item()

    @torch.no_grad
    def pred(self, X, device='cuda'):
        self.to(device)
        X = X.to(device)
        return self(X)


class LinearProbe(Probe):
    def __init__(self, input_size : int, output_size : int, **probe_kwargs):
        super(LinearProbe, self).__init__(**probe_kwargs)
        self.linear = torch.nn.Linear(input_size, output_size)

    
    def forward(self, x):
        return self.linear(x)

class MLPProbe(Probe):
    def __init__(self, num_layers : int, input_size : int, hidden_size : int, output_size : int, **probe_kwargs):
        super(MLPProbe, self).__init__(**probe_kwargs)
        modules = [
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            modules += [
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU()
            ]
        modules += [torch.nn.Linear(hidden_size, output_size)]
        self.model = torch.nn.ModuleList(modules)

    def forward(self, x):
        for module in self.model:
            x = module(x)
        return x


class AttentionProbe(Probe):
    """
    Attention probe for transformer activations with lower dimensional projection.
    Uses multi-head attention to aggregate sequence information for prediction.

    Args:
        input_size: The dimension of the input activations
        d_proj: The dimension of the projection
        nhead: The number of heads
        output_size: The dimension of the output
        max_length: (optional) The maximum length of the input sequence. Default is 8192.
        sliding_window: (optional) The sliding window size. Default is None.
        **probe_kwargs: Additional keyword arguments for the probe
    Returns:
        The output of the attention probe
    """

    def __init__(
        self,
        input_size: int,
        d_proj: int,
        nhead: int,
        output_size: int,
        max_length: int = 8192,
        sliding_window: Optional[int] = None,
        **probe_kwargs,
    ):
        super(AttentionProbe, self).__init__(**probe_kwargs)
        self.input_size = input_size
        self.d_proj = d_proj
        self.nhead = nhead
        self.q_proj = torch.nn.Linear(input_size, d_proj * nhead)
        self.k_proj = torch.nn.Linear(input_size, d_proj * nhead)
        self.v_proj = torch.nn.Linear(input_size, d_proj * nhead)
        self.out_proj = torch.nn.Linear(d_proj * nhead, output_size)

        if sliding_window is not None:
            mask = self._construct_sliding_window_mask(max_length, sliding_window)
        else:
            mask = self._construct_causal_mask(max_length)
        self.register_buffer("mask", mask)
        self.to(probe_kwargs["device"])

    def _construct_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Construct a causal (lower triangular) attention mask."""
        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask, diagonal=0)
        return mask.to(dtype=torch.bool)

    def _construct_sliding_window_mask(
        self, seq_len: int, window_size: int
    ) -> torch.Tensor:
        """Construct a sliding window causal attention mask."""
        q_idx = torch.arange(seq_len).unsqueeze(1)
        kv_idx = torch.arange(seq_len).unsqueeze(0)
        causal_mask = q_idx >= kv_idx
        windowed_mask = q_idx - kv_idx < window_size
        return causal_mask & windowed_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention probe.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len) or (seq_len,)
        """
        # Handle 2D input (seq_len, d_model) by adding batch dimension
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multi-head attention
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )

        # Apply scaled dot-product attention with mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask[:seq_len, :seq_len]
        )

        # Reshape back and project to output
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        output = self.out_proj(attn_output).squeeze(-1)

        if squeeze_output:
            output = output.squeeze(0)

        return output


class ProbeCV:
    def __init__(self, probe_class, n_split : int = 5, **probe_kwargs):
        self.probe_class = probe_class
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
            probe = self.probe_class(**self.probe_kwargs)
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


def get_activations(model : PreTrainedModel, X : dict, layer : int, batch_size : Optional[int] = None, efficient_mode : bool = False) -> torch.Tensor:
    """
    Extract activations from a specific layer of the model.
    
    Args:
        model: The pretrained model
        X: Dictionary with 'input_ids' and 'attention_mask'
        layer: The layer index to extract activations from
        batch_size: If provided, process in batches
        efficient_mode: If True, use forward hooks to capture only the target layer's 
                       activations (saves GPU memory by not storing all hidden states)
    """
    model.eval()

    if efficient_mode:
        # Use forward hooks to capture only the specified layer's activations
        captured_activations = []
        
        def activation_hook(module, input, output):
            # output is typically (batch_size, seq_len, hidden_dim) or a tuple
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            captured_activations.append(hidden_states.detach().float().cpu())
        
        # Register hook on the target layer
        hook_handle = model.model.layers[layer].register_forward_hook(activation_hook)
        
        try:
            if batch_size is not None:
                activations = []
                for b in trange(0, len(X['input_ids']), batch_size, desc="Collecting activations (efficient)..."):
                    captured_activations.clear()
                    batch_inputs = {
                        'input_ids': X['input_ids'][b:b + batch_size].to(model.device),
                        'attention_mask': X['attention_mask'][b:b + batch_size].to(model.device)
                    }
                    with torch.no_grad():
                        model(**batch_inputs, output_hidden_states=False)
                    activations.append(captured_activations[0])
                return torch.cat(activations, dim=0)
            else:
                with torch.no_grad():
                    model(**X, output_hidden_states=False)
                return captured_activations[0].squeeze()
        finally:
            hook_handle.remove()
    
    # Original method: output_hidden_states=True (stores all layer activations)
    if batch_size is not None:
        activations = []
        for b in trange(0, len(X['input_ids']), batch_size, desc="Collecting activations..."):
            batch_inputs = {
                'input_ids': X['input_ids'][b:b + batch_size].to(model.device),
                'attention_mask': X['attention_mask'][b:b + batch_size].to(model.device)
            }
            with torch.no_grad():
                batch_outputs = model(**batch_inputs, output_hidden_states=True)
                batch_activations = batch_outputs.hidden_states[layer].detach().float().cpu()
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
