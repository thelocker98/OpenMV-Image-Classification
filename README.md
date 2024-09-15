# OpenMV Classification Training

This repository contains a Jupyter notebook that demonstrates how to perform post-training integer quantization on machine learning models. This technique is particularly useful for reducing model size and improving inference speed, especially on low-power devices like the [OpenMV](https://openmv.io) camera.

## Features

- **Post-training Integer Quantization**: Optimize a model by converting 32-bit floating-point numbers to 8-bit fixed-point numbers.
- **Low-Power Device Compatibility**: The notebook is tailored for devices with limited computational resources, such as the OpenMV camera.
- **Efficient Model Deployment**: The techniques demonstrated ensure smaller model sizes and faster inference.

## Getting Started

### Setup

1. Clone the repository:
```bash
git clone https://github.com/thelocker98/openmv-classification-training.git cd openmv-classification-training
```
3. Run the notebook using Jupyter
```bash
jupyter lab
```

## How to Use
- **Open the Notebook**: Launch the notebook and execute all code cells in sequence.
- **Use Your Own Images**: Follow the steps outlined in the notebook to use your own images for training or testing.
- **Deploy the Model**:
    - Copy the quantized model from the `models` folder and load it onto the OpenMV camera.
    - Transfer the `boot.py` and `labels.txt` files from the `OpenMV code` folder to the camera.
    - Unplug and reconnect the camera to automatically run the model.
    
## References
- [OpenMV Camera](https://openmv.io)
- [Google post on model quantization](https://ai.google.dev/edge/litert/models/post_training_integer_quant)