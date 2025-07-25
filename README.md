 # DPCON:Distortion-Perception Co-Optimization Network for Distributed Image Compression

## Setup
### Environment
* `Ubuntu 22.04.5 LTS`
* `Python 3.8.8`
* `PyTorch 1.8.0+cu111`

### Installation

```shell
conda env create --file environment.yml
conda activate dpcon
```
### Dataset
The datasets used for experiments are KITTI Stereo and Cityscape.

For KITTI Stereo you can download the necessary image pairs from [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow_multiview.zip) and [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip). After obtaining `data_stereo_flow_multiview.zip` and `data_scene_flow_multiview.zip`, run the following commands:
```bash
unzip data_stereo_flow_multiview.zip # KITTI 2012
mkdir data_stereo_flow_multiview
mv training data_stereo_flow_multiview
mv testing data_stereo_flow_multiview

unzip data_scene_flow_multiview.zip # KITTI 2015
mkdir data_scene_flow_multiview
mv training data_scene_flow_multiview
mv testing data_scene_flow_multiview
```

For Cityscape you can download the image pairs from [here](https://www.cityscapes-dataset.com/downloads/). After downloading `leftImg8bit_trainvaltest.zip` and `rightImg8bit_trainvaltest.zip`, run the following commands:
```bash
mkdir cityscape_dataset
unzip leftImg8bit_trainvaltest.zip
mv leftImg8bit cityscape_dataset
unzip rightImg8bit_trainvaltest.zip
mv rightImg8bit cityscape_dataset
```
### Train DPCON
```shell
# train VAE-MFD In module/trainer_froze.py, freeze parameters and only train VAE-MFD function parameters
python train.py

# train PO-PFI In module/trainer_froze.py, freeze VAE-MFD parameters and only train PO-PFI parameters
python train.py

# train ARCO In module/trainer_froze.py, freeze both VAE-MFD and PO-PFI, and only train the adaptive weight map in ARCO
python train.py 
```



## Qualitative Results
### Cityscapes
<table align="center">
  <tr>
    <td align="center"><img src="images/c_6.png" width="200"><br><b>Ground Truth (bpp, PSNR&#8593, LPIPS&#8595)</b></td>
    <td align="center"><img src="images/c_ndic_6.png" width="200"><br><b>NDIC (0.041bpp, 27.60dB&#8593,  <br>0.2711&#8595)</b></td>
    <td align="center"><img src="images/c_ldmic_6.png" width="200"><br><b>LDMIC (0.037bpp, 26.65dB&#8593, 0.2850&#8595)</b></td>
  </tr>
  <tr>
    <td align="center"><img src="images/c_LD_6.png" width="200"><br><b>VAE-MFD (0.033bpp, 33.45dB&#8593, 0.1490&#8595)</b></td>
    <td align="center"><img src="images/c_LP_6.png" width="200"><br><b>VAE-MFD + PO-PFI (0.033bpp, 22.83dB&#8593, 0.1262&#8595)</b></td>
    <td align="center"><img src="images/c_alpha_6.png" width="200"><br><b>DPCON (0.033bpp, 31.98dB&#8593, 0.1036&#8595)</b></td>
  </tr>
</table>

### KITTI Stereo
<table align="center">
  <tr>
    <td align="center"><img src="images/k_495.png" width="200"><br><b>Ground Truth (bpp, PSNR&#8593, LPIPS&#8595)</b></td>
    <td align="center"><img src="images/k_ndic_495.png" width="200"><br><b>NDIC (0.084bpp, 21.46dB&#8593, <br>0.3984&#8595)</b></td>
    <td align="center"><img src="images/k_ldmic_495.png" width="200"><br><b>LDMIC (0.098bpp, 22.50dB&#8593, 0.3421&#8595)</b></td>
  </tr>
  <tr>
    <td align="center"><img src="images/k_LD_495.png" width="200"><br><b>VAE-MFD (0.081bpp, 22.78dB&#8593, 0.1689&#8595)</b></td>
    <td align="center"><img src="images/k_LP_495.png" width="200"><br><b>VAE-MFD + PO-PFI (0.081bpp, 20.62dB&#8593, 0.1328&#8595)</b></td>
    <td align="center"><img src="images/k_alpha_495.png" width="200"><br><b>DPCON (0.081bpp, 22.51dB&#8593, 0.1264&#8595)</b></td>
  </tr>
</table>







