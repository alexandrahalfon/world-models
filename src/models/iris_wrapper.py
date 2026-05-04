"""IRIS PyTorch wrapper using the official repo at vendor/iris/.

Stateful interface: call reset(initial_obs) once per trajectory, then step(action)
for each subsequent step. This matches IRIS's WorldModelEnv design (KV-cache).

IRIS operates at 64x64 RGB. Caller must pass num_actions matching the game.
The act(obs) method exposes IRIS's trained actor-critic policy for action collection.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_IRIS_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "vendor", "iris", "src"))
_IRIS_CFG = os.path.abspath(os.path.join(_HERE, "..", "..", "vendor", "iris", "config"))
if _IRIS_SRC not in sys.path:
    sys.path.insert(0, _IRIS_SRC)


class IRISWrapper:
    IMG_SIZE = 64

    def __init__(self, checkpoint_path: str, num_actions: int, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.num_actions = int(num_actions)
        self.agent = None
        self.wm_env = None

    def load_checkpoint(self) -> None:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from agent import Agent  # type: ignore
        from models.world_model import WorldModel  # type: ignore
        from models.actor_critic import ActorCritic  # type: ignore
        from models.transformer import TransformerConfig  # type: ignore
        from envs.world_model_env import WorldModelEnv  # type: ignore

        cfg_tok = OmegaConf.load(os.path.join(_IRIS_CFG, "tokenizer", "default.yaml"))
        cfg_wm = OmegaConf.load(os.path.join(_IRIS_CFG, "world_model", "default.yaml"))
        cfg_ac = OmegaConf.load(os.path.join(_IRIS_CFG, "actor_critic", "default.yaml"))

        tokenizer = instantiate(cfg_tok)

        wm_kwargs = {k: v for k, v in OmegaConf.to_container(cfg_wm, resolve=True).items()
                     if k != "_target_"}
        wm_config = TransformerConfig(**wm_kwargs)
        world_model = WorldModel(
            obs_vocab_size=tokenizer.vocab_size,
            act_vocab_size=self.num_actions,
            config=wm_config,
        )

        ac_kwargs = {k: v for k, v in OmegaConf.to_container(cfg_ac, resolve=True).items()
                     if k != "_target_"}
        actor_critic = ActorCritic(**ac_kwargs, act_vocab_size=self.num_actions)

        self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
        self.agent.load(self.checkpoint_path, self.device)
        self.agent.eval()

        self.wm_env = WorldModelEnv(
            tokenizer=self.agent.tokenizer,
            world_model=self.agent.world_model,
            device=self.device,
            env=None,
        )

    @staticmethod
    def _to_64(obs_uint8: np.ndarray) -> np.ndarray:
        if obs_uint8.shape[:2] == (IRISWrapper.IMG_SIZE, IRISWrapper.IMG_SIZE):
            return obs_uint8
        return np.array(Image.fromarray(obs_uint8).resize(
            (IRISWrapper.IMG_SIZE, IRISWrapper.IMG_SIZE), Image.BILINEAR))

    @torch.no_grad()
    def reset(self, obs_uint8: np.ndarray) -> np.ndarray:
        assert self.wm_env is not None, "Call load_checkpoint() first"
        obs64 = self._to_64(obs_uint8)
        x = torch.from_numpy(obs64).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(self.device)
        decoded = self.wm_env.reset_from_initial_observations(x)
        return (decoded.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    @torch.no_grad()
    def step(self, action: int) -> np.ndarray:
        assert self.wm_env is not None
        act_t = torch.tensor([int(action)], dtype=torch.long, device=self.device)
        obs, _, _, _ = self.wm_env.step(act_t, should_predict_next_obs=True)
        return (obs.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    def reset_actor(self, batch_size: int = 1) -> None:
        """Initialize the actor-critic's LSTM hidden state. Call once per trajectory.

        IRIS's ActorCritic.__init__ sets self.hx = self.cx = None; the LSTM forward
        in actor_critic.py:91 fails on None. ActorCritic.reset(n) zeros them out.
        """
        assert self.agent is not None, "Call load_checkpoint() first"
        self.agent.actor_critic.reset(n=batch_size)

    @torch.no_grad()
    def act(self, obs_uint8: np.ndarray, temperature: float = 1.0,
            should_sample: bool = True) -> int:
        """Sample an action from IRIS's trained actor-critic given a uint8 frame.

        Auto-initializes the LSTM hidden state on first call (or after reset_actor
        cleared it) so callers don't have to remember.
        """
        assert self.agent is not None, "Call load_checkpoint() first"
        ac = self.agent.actor_critic
        if getattr(ac, "hx", None) is None or getattr(ac, "cx", None) is None:
            ac.reset(n=1)
        obs64 = self._to_64(obs_uint8)
        x = torch.from_numpy(obs64).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(self.device)
        act_t = self.agent.act(x, should_sample=should_sample, temperature=temperature)
        return int(act_t.item())

    def predict(self, obs, action):
        raise NotImplementedError("IRIS is stateful. Use reset(obs) once + step(action) per step.")
