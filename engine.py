import modal
from mi_observatory.api.handlers import (
    handle_analyze_request,
    handle_status_request,
)
from mi_observatory.services.model_runtime import ModelRuntime

# 1. Define the Environment
# We need transformer_lens for the dissection and fastapi for the URL endpoints.
image = (
    modal.Image.debian_slim()
    .pip_install("transformer_lens", "torch", "scikit-learn", "numpy", "fastapi[standard]")
    .add_local_python_source("mi_observatory"))
app = modal.App("mi-observatory", image=image)

@app.cls(gpu="any", scaledown_window=120)
class Model:
    @modal.enter()
    def ignition(self):
        """IGNITION: Initialize runtime and load model into VRAM once."""
        self.runtime = ModelRuntime()
        self.runtime.load_model()

    @modal.fastapi_endpoint(method="POST")
    def analyze(self, data: dict):
        """API endpoint: returns inspect snapshot for requested components."""
        return handle_analyze_request(self.runtime, data)

    @modal.fastapi_endpoint(method="GET")
    def status(self):
        """Sanity Check: Confirms the container is reachable."""
        return handle_status_request(self.runtime)

        