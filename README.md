 # DPCON:Distortion-Perception Co-Optimization Network for Distributed Image Compression

**Note**: We currently provide partial experimental results and visualizations on Cityscapes datasets. The full codebase and pre-trained weights will be released upon paper acceptance.
## Experiment result on Cityscapes dataset
![city](https://github.com/user-attachments/assets/69f4879f-bcbc-4d8f-a489-1a008717560c)
## Qualitative Results
### Cityscapes
<table align="center">
  <tr>
    <td align="center"><img src="images/c_6.png" width="200"><br><b>Ground Truth (bpp, PSNR, LPIPS)</b></td>
    <td align="center"><img src="images/c_ndic_6.png" width="200"><br><b>NDIC (0.041bpp, 27.60dB,   0.2711)</b></td>
    <td align="center"><img src="images/c_ldmic_6.png" width="200"><br><b>LDMIC (0.037bpp, 26.65dB, 0.2850)</b></td>
  </tr>
  <tr>
    <td align="center"><img src="images/c_LD_6.png" width="200"><br><b>VAE-MFD (0.033bpp, 33.45dB, 0.1490)</b></td>
    <td align="center"><img src="images/c_LP_6.png" width="200"><br><b>VAE-MFD + PO-PFI (0.033bpp, 22.83dB, 0.1262)</b></td>
    <td align="center"><img src="images/c_alpha_6.png" width="200"><br><b>DPCON (0.033bpp, 31.98dB, 0.1036)</b></td>
  </tr>
</table>

### KITTI Stereo
<table align="center">
  <tr>
    <td align="center"><img src="images/k_135.png" width="200"><br><b>Ground Truth (bpp, PSNR, LPIPS)</b></td>
    <td align="center"><img src="images/k_ndic_135.png" width="200"><br><b>NDIC (0.081bpp, 21.70dB, 0.3627)</b></td>
    <td align="center"><img src="images/k_ldmic_135.png" width="200"><br><b>LDMIC (0.099bpp, 22.41dB, 0.3194)</b></td>
  </tr>
  <tr>
    <td align="center"><img src="images/k_LD_135.png" width="200"><br><b>VAE-MFD (0.078bpp, 22.95dB, 0.2528)</b></td>
    <td align="center"><img src="images/k_LP_135.png" width="200"><br><b>VAE-MFD + PO-PFI (0.078bpp, 21.02dB, 0.2359)</b></td>
    <td align="center"><img src="images/k_alpha_135.png" width="200"><br><b>DPCON (0.078bpp, 22.72dB, 0.2347)</b></td>
  </tr>
</table>







