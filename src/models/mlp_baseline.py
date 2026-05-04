"""MLP world model baseline operating on Atari 128-byte RAM state (not pixels).

A 3-layer hidden-256 MLP cannot learn in 84x84x3=21168-dim pixel space, so this baseline
uses the RAM observation space instead. Its error-growth alpha is not directly comparable
to pixel-space models (IRIS, DIAMOND, DreamerV3); see CLAUDE.md.

Provides the unified reset(obs) + step(action) interface used by rollout.py, plus a
stateless predict(obs, action) for backward compatibility.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPWorldModel(nn.Module):
    def __init__(self, input_dim: int = 129, output_dim: int = 128,
                 hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ram_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ram_state.float(), action.float().unsqueeze(-1)], dim=-1)
        return self.net(x)


def _best_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPBaseline:
    def __init__(self, hidden_dim: int = 256, n_layers: int = 3,
                 lr: float = 1e-3, device: str = "cuda"):
        self.device = _best_device(device)
        self.model = MLPWorldModel(hidden_dim=hidden_dim, n_layers=n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self._cur_obs: np.ndarray | None = None

    def train(self, ram_states: np.ndarray, actions: np.ndarray, next_ram_states: np.ndarray,
              n_steps: int = 200_000, batch_size: int = 256) -> list:
        s = torch.tensor(ram_states / 255.0, dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.float32)
        ns = torch.tensor(next_ram_states / 255.0, dtype=torch.float32)
        dataset = TensorDataset(s, a, ns)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        losses = []
        step = 0
        while step < n_steps:
            for s_b, a_b, ns_b in loader:
                s_b, a_b, ns_b = s_b.to(self.device), a_b.to(self.device), ns_b.to(self.device)
                pred = self.model(s_b, a_b)
                loss = self.loss_fn(pred, ns_b)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                step += 1
                if step >= n_steps:
                    break
        return losses

    def predict(self, ram_state: np.ndarray, action: int) -> np.ndarray:
        """Stateless single-step prediction. ram_state: [128] uint8, action: scalar int."""
        self.model.eval()
        with torch.no_grad():
            s = torch.tensor(ram_state / 255.0, dtype=torch.float32).unsqueeze(0).to(self.device)
            a = torch.tensor([action], dtype=torch.float32).to(self.device)
            pred = self.model(s, a).squeeze(0).cpu().numpy()
        return (pred * 255.0).clip(0, 255).astype(np.uint8)

    # Unified stateful interface (matches IRISWrapper / DIAMONDWrapper)
    def reset(self, obs_uint8: np.ndarray) -> np.ndarray:
        self._cur_obs = obs_uint8.astype(np.uint8).copy()
        return self._cur_obs.copy()

    def step(self, action: int) -> np.ndarray:
        assert self._cur_obs is not None, "Call reset(obs) first"
        nxt = self.predict(self._cur_obs, int(action))
        self._cur_obs = nxt
        return nxt.copy()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
