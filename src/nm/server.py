from pathlib import Path

import litserve as ls
import torch
from litserve import Request, Response

from nm.config import AppConfig
from nm.module import ToxicModel


class SimpleLitAPI(ls.LitAPI):
    def setup(
        self,
        device: str,
        checkpoint: str | Path,
    ) -> None:
        self.device = device
        self.checkpoint = (
            Path(checkpoint) if isinstance(checkpoint, str) else checkpoint
        )
        self.precision = torch.bfloat16

        self.lit_module = ToxicModel.load_from_checkpoint(self.checkpoint)
        self.lit_module.to(self.device).to(self.precision)
        self.lit_module.eval()

    async def decode_request(self, request: Request):
        return request["input"]

    async def predict(self, input: str) -> torch.Tensor:
        return self.lit_module.predict_step(input)

    async def encode_response(self, output: torch.Tensor) -> Response:
        return {
            label: prob.item() for label, prob in zip(self.labels, output, strict=False)
        }


def main() -> None:
    api = SimpleLitAPI(enable_async=True)
    server = ls.LitServer(
        api,
        accelerator=AppConfig.accelerator,
        devices=AppConfig.devices,
        timeout=AppConfig.timeout,
        track_requests=AppConfig.track_requests,
    )
    server.run(port=8000, generate_client_file=AppConfig.generate_client_file)
