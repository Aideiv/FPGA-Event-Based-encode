<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Learning Normal Flow Directly from Event Neighborhoods </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://github.com/dhyuan99" target="_blank" style="text-decoration: none;">Dehao Yuan</a>&nbsp;&nbsp;
    <a href="https://www.aftersomemath.com" target="_blank" style="text-decoration: none;">Levi Burner</a>&nbsp;&nbsp;
    <a href="https://jiayi-wu-leo.github.io" target="_blank" style="text-decoration: none;">Jiayi Wu</a>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=UKAsIsUAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Minghui Liu</a>&nbsp;&nbsp;
    <a href="https://codingrex.github.io" target="_blank" style="text-decoration: none;">Jingxi Chen</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Yiannis Aloimonos</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~fer/" target="_blank" style="text-decoration: none;">Cornelia Fermüller</a>
    <br><br>
    <a href="https://prg.cs.umd.edu" target="_blank" style="text-decoration: none;">Perception and Robotics Group at University of Maryland</a>
    <br><br>
    <a href="https://arxiv.org/abs/2412.11284" target="_blank" style="text-decoration: none;">[Paper]</a> &nbsp;&nbsp
    <a href="https://drive.google.com/drive/folders/1gkmUyZX5VRf8DxiBKL9CSdWdifjqZVq3?usp=sharing" target="_blank" style="text-decoration: none;">[Flow Prediction Videos (42 scenes from 5 datasets)]</a>
</p>

## News
🚀 This work has been accepted by ICCV2025! See you in Honolulu, Hawaii!

🚀 A real-time implementation of this normal flow estimator: [https://github.com/dhyuan99/VecKM_flow_cpp](https://github.com/dhyuan99/VecKM_flow_cpp).

## Abstract
<div align="center">
<img src="assets/abstract.png">
</div>

<div align="center">
<img src="assets/normal_flow.png">
</div>

## API Usage
It is a generic normal flow estimator with event camera inputs. The API is easily used by following codes, after [installing the package](#installation). See [demo](./demo/) for the codes and data for running the demo:
```
cd demo
python main.py
```
The API call is as simple as followed:
``` python
from models.inference import NormalFlowEstimator
from models.visualize import gen_flow_video

estimator = NormalFlowEstimator()
flow_predictions, flow_uncertainty = estimator.inference(events_t, undistorted_events_xy)
flow_predictions[flow_uncertainty > 0.3] = np.nan

gen_flow_video(
    events_t.numpy(), 
    undistorted_events_xy.numpy(), 
    flow_predictions.numpy(), 
    './frames', './output.mp4', fps=30)
```

The data dimensions are as followed:
| Variables        | Description | Data Dimension  |
|-------------|-----|-------------|
| `events_t`  | Sorted event time in seconds | `(n, )` float64    |
| `undistorted_events_xy` | Undistorted normalized event coordinates (focal length one). The range shall be around (-1, 1). See [Undistorted Normalized Coordinates](#undistorted-normalized-coordinates) for computing them. 1st row is width, 2nd row is height.  | `(n, 2)` float32      |
| `flow_predictions` | Predicted normal flow. Unit: undistorted normalized pixels per second. | `(n, 2)` float32      |
| `flow_uncertainty` | Prediction uncertainty. | `(n, )` float32 >= 0 |

The prediction is visualized as a video like this:
<div align="center">
<img src="assets/demo.gif" alt="Watch the video" width="100%">
</div>

#### Training Set Options
We provide four estimators, which can be selected by the parameter `training_set`. The four possible parameters are `"UNION"`, `"MVSEC"`, `"DSEC"`, `"EVIMO"`, which state the training set that the estimator is trained on. `"UNION"` is the model that trained on the union of the three datasets. Therefore, it is generally recommended to use `training_set="UNION"`, which is set as default.
``` python
estimator = NormalFlowEstimator(training_set="UNION")  # default model
```

## Installation
```
git clone <your-repo-url>
cd <repo-name>

conda create -n event-flow python=3.11
conda activate event-flow

pip install --upgrade pip setuptools wheel
python setup.py sdist bdist_wheel
pip install .
```
If you encounter any issues installing, please raise an issue.

## Undistorted Normalized Coordinates
To obtain the undistorted normalized coordinates, one needs to utilize `cv2.undistortPoints` and obtain the intrinsic camera matrix `K` and distortion coefficient `D` from the dataset.
``` python
def get_undistorted_events_xy(raw_events_xy, K, D):
    # raw_events_xy has shape (n, 2). The range is e.g. (0, 640) int X (0, 480) int.
    raw_events_xy = raw_events_xy.astype(np.float32)
    undistorted_normalized_xy = cv2.undistortPoints(raw_events_xy.reshape(-1, 1, 2), K, D)
    undistorted_normalized_xy = undistorted_normalized_xy.reshape(-1, 2)
    return undistorted_normalized_xy
```

## Evaluated Datasets
**[Recommend to Watch]** We evaluated the estimator on [MVSEC](https://daniilidis-group.github.io/mvsec/), [DSEC](https://dsec.ifi.uzh.ch), [EVIMO](https://better-flow.github.io/evimo/download_evimo_2.html), [FPV](https://fpv.ifi.uzh.ch), [VECtor](https://star-datasets.github.io/vector/). The flow prediction videos of every evaluated scene (in total 42 scenes) can be found here: 

<div align="center">
    <a href="https://drive.google.com/drive/folders/1gkmUyZX5VRf8DxiBKL9CSdWdifjqZVq3?usp=sharing" target="_blank">
    <img src="assets/video_icon.png" alt="Watch the video" width="300">
    </a>
</div>

**[Reproduce the inference]** We precompute the undistorted normalized coordinates. They can be downloaded in [this drive](https://drive.google.com/drive/folders/1M7vaokRF71f91AtWBaQuZlSvOiTSePox?usp=sharing). The data format is exactly the same as the [demo data](demo/demo_data). Therefore, it is straight-forward to use the API to inference the datasets.

## Egomotion Estimation
We propose an SVM-based egomotion estimator in the paper. The estimator uses predicted normal flow and IMU rotational measurement to predict the translation direction. The implementation is in [`./egomotion`](./egomotion).

<div align="center">
<img src="assets/egomotion.gif" alt="Watch the video" width="100%">
</div>

## Train Your Own Estimator
Look at [this](train).

## Citations
If you find this helpful, please consider citing
```
@article{yuan2024learning,
  title={Learning Normal Flow Directly From Event Neighborhoods},
  author={Yuan, Dehao and Burner, Levi and Wu, Jiayi and Liu, Minghui and Chen, Jingxi and Aloimonos, Yiannis and Ferm{\"u}ller, Cornelia},
  journal={arXiv preprint arXiv:2412.11284},
  year={2024}
}
```


---

## Drone Collision Avoidance (FPGA + ARM)

This repo has been extended with a **real-time drone collision avoidance system** that runs the normal flow estimator on **FPGA** with an **ARM co-processor** for flight control.

### Architecture

```
Event Camera (AER) → [FPGA] → Flow Vectors → [ARM] → Motor Commands
                    ╰─────────┬─────────╯ ╰──────┬──────╯
                       encoder_systolic     collision_predictor
                       spatial_hash         evasion_controller
                       ring_buf / norm       safety_watchdog
```

### Quick Start (Software Simulation)

```bash
# 1. Convert pretrained weights (d=384 → d=128 for FPGA)
python train/convert_weights_to_fpga.py \
    --input models/models/UNION.pth \
    --output models/models/FPGA.pth

# 2. Run the end-to-end simulation (no hardware needed)
python test/test_fpga_simulator.py --visualize
```

### Directory Layout

| Directory | Purpose |
|-----------|---------|
| [`fpga/`](./fpga/) | HLS/C++ FPGA modules: AER interface, ring buffer, normalization, spatial hash k-NN, systolic encoder array, PWM output |
| [`arm/`](./arm/) | ARM C++ controller: collision predictor (TTC & clustering), evasion controller (potential fields), safety watchdog, main control loop |
| [`test/`](./test/) | Unit tests: Python FPGA pipeline simulator (`test_fpga_simulator.py`), ARM C++ unit tests (`test_arm_collision_predictor.cpp`), HLS C-simulation testbench (`testbench.cpp`) |
| [`train/`](./train/) | Training configs: `fpga_training_config.py` for d=128 model, `convert_weights_to_fpga.py` |
| [`drone/`](./drone/) | Python drone control package (for HIL testing on companion computer) |
| [`models/`](./models/) | `FPGAParams` in `params.py`, k-NN `NormalEstimator` in `estimator.py`, `inference.py` with FPGA mode |

### Key Production-Readiness Features

- **Safety Watchdog** → RC link failsafe, altitude ceiling, NaN detection, motor timeout
- **Hysteresis** → Prevents evasion-level thrashing (EMERGENCY → WARNING → NONE with hold times)
- **Configurable** → Central `fpga/config.yaml` mirrors constants across all layers
- **Test Coverage** → Python simulation + C++ unit tests + HLS testbench
- **FPGA Weights Export** → `convert_weights_to_fpga.py --export-fpga-header` generates `encoder_weights.h`

### Testing

```bash
# Python pipeline simulation (5 tests: looming, lateral, noise, multi, throughput)
python test/test_fpga_simulator.py

# ARM C++ unit tests
cd arm && make test && ./test_arm_cp

# HLS C-simulation (requires Vitis HLS)
cd fpga && vitis_hls -f build.tcl
```

This project is an extension project from [VecKM](https://github.com/dhyuan99/VecKM), an ICML2024 paper.
```
@InProceedings{pmlr-v235-yuan24b,
  title = 	 {A Linear Time and Space Local Point Cloud Geometry Encoder via Vectorized Kernel Mixture ({V}ec{KM})},
  author =       {Yuan, Dehao and Fermuller, Cornelia and Rabbani, Tahseen and Huang, Furong and Aloimonos, Yiannis},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {57871--57886},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/yuan24b/yuan24b.pdf},
  url = 	 {https://proceedings.mlr.press/v235/yuan24b.html},
}