import onnxruntime as ort
import torch
from nn_model import Net  # Import the model class for reference

def validate_onnx_model():
    # Load the PyTorch model (for reference in case you want to compare results)
    model = Net()
    model.load_state_dict(torch.load("/app/model.pth"))
    model.eval()  # Set the model to evaluation mode

    try:
        # Load the ONNX model using ONNX Runtime
        onnx_model_path = "app/model.onnx"
        session = ort.InferenceSession(onnx_model_path)

        # Get model input details
        inputs = session.get_inputs()
        input_shape = inputs[0].shape  # Should be [None, 2] for your model

        # Log input and output details for reference
        print(f"Model Inputs: {[i.name for i in inputs]} with shape {input_shape}")
        print(f"Model Outputs: {[o.name for o in session.get_outputs()]}")

        # Ensure the model expects exactly 2 features as per your PyTorch model
        if len(input_shape) != 2 or input_shape[1] != 2:
            raise ValueError(f"Model input shape mismatch: Expected [None, 2], got {input_shape}")

        # Dummy input for testing: (batch_size=1, features=2)
        dummy_input = torch.tensor([[0.5, 0.5]], dtype=torch.float32).unsqueeze(0)

        # Perform inference using PyTorch model (for comparison)
        with torch.no_grad():
            pytorch_result = model(dummy_input)

        print(f"PyTorch Model Inference Result: {pytorch_result}")

        # Perform inference using ONNX model
        onnx_result = session.run(None, {inputs[0].name: dummy_input.numpy()})
        print(f"ONNX Model Inference Result: {onnx_result}")

        print("ONNX model validation successful!")

    except Exception as e:
        print(f"ONNX model validation failed: {e}")
        exit(1)

if __name__ == "__main__":
    validate_onnx_model()
