"""DIAMOND PyTorch wrapper using the official repo at vendor/diamond/.

Stateful interface: reset(initial_obs) once, step(action) per step.
DIAMOND requires 4 conditioning frames + 4 conditioning actions; the wrapper
maintains a sliding deque internally. Initial obs is replicated 4x at reset.

Operates at 64x64 RGB in [-1, 1] internally; reset/step use uint8 [0, 255].
"""
from __future__ import annotations
import os
import sys
from collections import deque
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIAMOND_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "vendor", "diamond", "src"))
if _DIAMOND_SRC not in sys.path:
    sys.path.insert(0, _DIAMOND_SRC)


class DIAMONDWrapper:
    IMG_SIZE = 64
    NUM_COND = 4
    NUM_DENOISING_STEPS = 3   # atari_100k DIAMOND default

    def __init__(self, checkpoint_path: str, agent_config_path: str,
                 num_actions: int, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.agent_config_path = agent_config_path
        self.num_actions = int(num_actions)
        self.agent = None
        self.sampler = None
        self._obs_buf: deque = deque(maxlen=self.NUM_COND)
        self._act_buf: deque = deque(maxlen=self.NUM_COND)

    def load_checkpoint(self) -> None:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from agent import Agent  # type: ignore
        from models.diffusion import DiffusionSampler, DiffusionSamplerConfig  # type: ignore

        raw = OmegaConf.load(self.agent_config_path)
        # The shipped agent_config.yaml uses absolute interpolations like
        #   ${agent.denoiser.inner_model.img_channels}
        #   ${env.train.size}
        # Wrap under a synthetic root so those refs resolve.
        wrapped = OmegaConf.create({
            "env": {"train": {"size": self.IMG_SIZE}},
            "agent": raw,
        })
        OmegaConf.resolve(wrapped)

        agent_cfg = instantiate(wrapped.agent, num_actions=self.num_actions)
        self.agent = Agent(agent_cfg).to(self.device)
        self.agent.load(self.checkpoint_path)
        self.agent.eval()

        sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=self.NUM_DENOISING_STEPS)
        self.sampler = DiffusionSampler(self.agent.denoiser, sampler_cfg)

    @staticmethod
    def _to_64(obs_uint8: np.ndarray) -> np.ndarray:
        if obs_uint8.shape[:2] == (DIAMONDWrapper.IMG_SIZE, DIAMONDWrapper.IMG_SIZE):
            return obs_uint8
        return np.array(Image.fromarray(obs_uint8).resize(
            (DIAMONDWrapper.IMG_SIZE, DIAMONDWrapper.IMG_SIZE), Image.BILINEAR))

    def _to_minus1_1(self, obs_64: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(obs_64).permute(2, 0, 1).float().div_(255.0)
        return (x.mul_(2.0).sub_(1.0)).to(self.device)

    @staticmethod
    def _to_uint8(t: torch.Tensor) -> np.ndarray:
        x = ((t.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).byte()
        return x.permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def reset(self, obs_uint8: np.ndarray) -> np.ndarray:
        assert self.agent is not None, "Call load_checkpoint() first"
        obs64 = self._to_64(obs_uint8)
        x = self._to_minus1_1(obs64)
        self._obs_buf = deque([x.clone() for _ in range(self.NUM_COND)], maxlen=self.NUM_COND)
        self._act_buf = deque([0 for _ in range(self.NUM_COND)], maxlen=self.NUM_COND)
        return self._to_uint8(x)

    @torch.no_grad()
    def step(self, action: int) -> np.ndarray:
        assert self.sampler is not None
        self._act_buf.append(int(action))
        prev_obs = torch.stack(list(self._obs_buf), dim=0).unsqueeze(0)  # (1,T,3,64,64)
        prev_act = torch.tensor([list(self._act_buf)], dtype=torch.long, device=self.device)
        next_obs, _ = self.sampler.sample(prev_obs, prev_act)  # (1,3,64,64) in [-1,1]
        next_obs = next_obs.squeeze(0)
        self._obs_buf.append(next_obs.clone())
        return self._to_uint8(next_obs)

    def predict(self, obs, action):
        raise NotImplementedError("DIAMOND is stateful. Use reset(obs) once + step(action) per step.")
