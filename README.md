# DPCON:Distortion-Perception Co-Optimization Network for Distributed Image Compression
## Experiment result on Cityscapes dataset
![city](https://github.com/user-attachments/assets/69f4879f-bcbc-4d8f-a489-1a008717560c)
## Qualitative Results

<style>
  .figure-container {
    display: flex;
    justify-content: center;
    gap: 20px; 
    margin-bottom: 30px;
  }
  figure {
    margin: 0;
    text-align: center;
  }
  figcaption {
    margin-top: 10px;
    font-weight: bold;
  }
</style>

<div class="figure-container">
  <figure>
    <img src="images/c_6.png" width="200">
    <figcaption>Ground Truth</figcaption>
  </figure>
  <figure>
    <img src="images/c_ndic_6.png" width="200">
    <figcaption>NDIC</figcaption>
  </figure>
  <figure>
    <img src="images/c_ldmic_6.png" width="200">
    <figcaption>LDMIC</figcaption>
  </figure>
</div>

<div class="figure-container">
  <figure>
    <img src="images/c_LD_6.png" width="200">
    <figcaption>VAE-MFD</figcaption>
  </figure>
  <figure>
    <img src="images/c_LP_6.png" width="200">
    <figcaption>VAE-MFD + PO-PFI</figcaption>
  </figure>
  <figure>
    <img src="images/c_alpha_6.png" width="200">
    <figcaption>DPCON</figcaption>
  </figure>
</div>



