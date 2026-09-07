"""ResNet50 with the post-hoc COVID corrector."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from covid_corrector.correctors.fisher import FisherCorrector
from covid_corrector.correctors.preprocessing import PCAPreprocessor
from covid_corrector.data.transforms import TransformSpec, build_transforms
from covid_corrector.models.resnet import load_resnet

SERVICE_DIR = Path(__file__).resolve().parent
CLASSES = ("NORMAL", "PNEUMONIA", "TURBERCULOSIS")
COVID_CLASS = "COVID19"


class ModelCorrected:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = load_resnet(SERVICE_DIR / "bin" / "legacy_resnet50.pt", device)
        self.preprocessor = PCAPreprocessor.load(
            SERVICE_DIR / "bin" / "full_hidden_preprocessor.joblib"
        )
        self.corrector = FisherCorrector.load(
            SERVICE_DIR / "bin" / "full_hidden_fisher_corrector.joblib"
        )
        self.transform = build_transforms(TransformSpec(input_size=224))

    def _hidden_modules(self):
        modules = [self.model.maxpool]
        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            modules.extend(getattr(self.model, stage_name))
        return modules

    def _extract(self, image: Image.Image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        outputs = {}
        hooks = []

        for index, module in enumerate(self._hidden_modules()):
            hooks.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, index=index: outputs.__setitem__(
                        index, output
                    )
                )
            )

        try:
            with torch.inference_mode():
                logits = self.model(tensor)
                hidden = torch.cat(
                    [
                        torch.nn.functional.adaptive_avg_pool2d(
                            outputs[index], (2, 2)
                        ).flatten(1)
                        for index in range(len(hooks))
                    ],
                    dim=1,
                )
        finally:
            for hook in hooks:
                hook.remove()

        return logits.cpu().numpy(), hidden.cpu().numpy()

    def inference(self, image: Image.Image) -> dict:
        logits, hidden = self._extract(image)
        legacy_index = int(np.argmax(logits[0]))
        legacy_prediction = CLASSES[legacy_index]

        features = self.preprocessor.transform(hidden)
        score = float(self.corrector.decision_function(features)[0])
        corrected = score >= self.corrector.threshold

        return {
            "legacy_prediction": legacy_prediction,
            "final_prediction": COVID_CLASS if corrected else legacy_prediction,
            "correction_applied": bool(corrected),
            "corrector_score": score,
            "corrector_threshold": float(self.corrector.threshold),
        }
