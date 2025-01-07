# A Gan-Based Approach to Erosion Simulation

Authors: David Sommerfield and Nicholas Moen

This repository contains code and a paper detailing a convolutional neural network (CNN) implementation for simulating hydraulic erosion on heightmaps. Designed for game development and real-time terrain authoring, the CNN achieves high performance and generates realistic terrain features efficiently.

Note: the paper was just made for a university course, so it's less polished than a typical academic publication. See our presentation here: [Google Slides Link](https://docs.google.com/presentation/d/1EYl_tC0HDfzUmDDKzc6oarilb0BzYCXmm8SSXUFyuRQ/edit?usp=sharing).

---

## Results
![image](https://github.com/user-attachments/assets/1d6bfeb7-8f7b-467d-afba-9696efae04af)

![image](https://github.com/user-attachments/assets/d17c4199-d9d8-49f0-acee-ab54c196dc1f)

![image](https://github.com/user-attachments/assets/213f36f2-7788-42f1-af63-806cdb23f9b4)


### Input

![image](https://github.com/user-attachments/assets/57a493c6-d8fe-4139-9810-70e98e4ad48c)


### Predicted

![image](https://github.com/user-attachments/assets/bf70288c-c247-453d-8775-4280cc57ffba)

### Actual

![image](https://github.com/user-attachments/assets/2ef888fc-dce5-4f9c-9ef7-3053c1a3e905)

---

## Key Features

- **Architecture**: Based on a modified pix2pix model, the framework incorporates encoder-decoder structures and a patch-based discriminator. The generator utilizes dilated convolution layers to capture long-range spatial patterns like sediment carry.
- **Performance**: This model performs signifigantly faster than traditional simulation (**~192x** faster than the simulation we used to generate the training data). On three Quadro RTX 6000/8000 GPUs, the model generates 32 terrain images in 0.1 seconds. Even a single GTX 1060 achieves this in 0.3 seconds, making it suitable for real-time applications.
- **Data Generalization**: Trained on a diverse set of procedurally generated heightmaps (e.g., Perlin noise), the model demonstrates adaptability by effectively simulating erosion on unseen data.
- **Limitations**: The model exhibits minor grid artifacts in the output terrain and can benefit from further refinements in detail generation.

## Datasets

The training data was created using a custom data acquisition pipeline, described in detail in the "Datasets" section of the final project report. 

- **Input Images**: Generated using Perlin noise maps with adjustable parameters such as scale, noise level, amplitudes, and variances all distrubted according to a normal distribution. These parameters enabled the creation of diverse terrain characteristics and helped avoid artifacts resulting from sparse input.
- **Output Images**: Created by running the input DEMs through a hydraulic erosion algorithm.

To optimize the data for GAN-based architectures, images were encoded as 16-bit `.tiff` files, allowing parameter values to range from 0–65535 instead of the typical 0-255 (important for smooth height representation). 

Encoding details:
- **Red Channel**: Height (the only value that differs between input and output).
- **Green Channel**: Rock hardness.
- **Blue Channel**: Erosion amount in simulation (constant per image).

![image](https://github.com/user-attachments/assets/794a98d2-9521-4655-b371-eb6918807add)

![image](https://github.com/user-attachments/assets/d5413940-0c85-4797-9770-1fa9df8df85b)


## Model Overview

### Generator
- **Encoder**: Downsampling with dilated convolutions to capture both local and long-distance features.
- **Decoder**: Transposed convolutions and dilated layers to upsample and refine the output.

### Discriminator
- Patch-based evaluation (patch size: 70×70) focuses on localized details for realistic texture reproduction.

### Training
- **Loss Function**: Mean Absolute Error (MAE) and LPIPS for realism and perceptual similarity.
- **Augmentation**: Random rotations to improve robustness across terrain orientations (not strictly necessary).
- **Hardware**: Trained on three Quadro RTX (6000/8000) GPUs over 200 epochs (21 hours).

---

## Setup

1. **Create Conda Environment**:
```bash
conda env create -f tfnumpy.yaml
```

2. **Activate Environment**:
```bash
conda activate tfnumpy
```

3. **Configure Dataset Path**: 
- Change the `dataPath` variable in `Codes/ErosionArchitecture/test/test_generator.py` (line 83) and `Codes/ErosionArchitecture/models/train.py` (line 92).

### Training
To start training:
1. Navigate to `models` directory:
```
cd Codes/ErosionArchitecture/models
```

2. Run the training Script
```bash
python3 train.py
```

### Testing

After training, run the testing script:
1. Navigate to `test` directory:
```bash
cd Codes/ErosionArchitecture/test
```

2. Run the test script:
```bash
python3 test_generator.py
```

This will generate output images (`inp_.png`, `generated_.png`, and `real_.png`) for evaluation.

### Custom Images

You can modify the `ErosionOnFileWith8_BitImages()` function in `test_generator.py` to use your own images (e.g., `test.png`, `blurredhllogo.png`). The outputs will be saved as `art_output.png` and `eroded_hllogo.png`.

### Dependencies

- **Python**: 3.11.3
- **Keras**: 2.4.3
- **Tensorflow**: 2.4.1
- **PyTorch**: 2.2.0
- **CUDA**: cudnn=7.6.5
- **NumPy**: 1.23.1

---
For further details, consult the provided paper or feel free to contact us.

