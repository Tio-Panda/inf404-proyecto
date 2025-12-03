---
marp: true
paginate: true
theme: beam
header: Deep Learning for Enhancing Ultrafast Ultrasound Image Reconstruction
footer: LACSC 2025
---

<!-- _class: title -->
<style scoped> h1{font-size:47px;}</style>

#  Deep Learning for Enhancing Ultrafast Ultrasound Image Reconstruction

<div style="position:relative; top: 90px; text-align: center">
Sebastián Gutiérrez, Ricardo Ñanculef, Hernan Mella,  <br>
Joaquín Mura, Julio Sotelo

<div style="text-align:center;">
</div>
<div style="margin-top: 20px">
</div>

<div style="margin-top: 5px">
</div>

</div>

<div style="position:absolute;top:10px;left:10px; width:200px;">
<img src="image-1.png">
</div>

<div style="position:absolute;top:10px;right:10px; width:200px;">
<img src="image-2.png">
</div>

---

# Table of Contents

<div class="beam-toc">
    
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Background</h3>
      </div>
  </div>
    
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Problem</h3>
      </div>
  </div>
  
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Solution</h3>
      </div>
  </div>

  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Results</h3>
      </div>
  </div>
  
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Discussion & Conclusions</h3>
      </div>
  </div>
</div>

---
# Background — Ultrasound

Ultrasound is an imaging modality that uses acoustic pulses to produce real-time images. It stands out for its safety (no radiation), portability and low cost compared with magnetic resonance imaging and computed tomography.

---
# Background — PICMUS and CUBDL

**Plane-wave Imaging Challenge in Medical UltraSound (PICMUS)** is an academic challenge focused on plane-wave imaging for ultrasound. It standardizes data, protocols, and metrics to enable objective comparisons of beamforming in plane-wave/ultrafast imaging.

**Challenge on Ultrasound Beamforming with Deep Learning (CUBDL)** is a challenge aimed at applying deep learning to ultrasound beamforming. It was created to explore, validate, and guide the use of deep learning methods that match or surpass classical approaches.

---
# Background — One deep learning application to ultrasound

**Beamforming-integrated neural networks for ultrasound imaging (Xiao et al., 2025)**: integrate a sparse matrix beamforming (SMB) layer into a CNN (BINN), explicitly encoding DAS. They achieve ~5 ms inference and a 9.8% MSLE improvement over a standard CNN, making it viable for ultrafast imaging.

---
# Background — Beamforming

Beamforming is the technique that combines signals from multiple transducer elements to “form” and steer a beam toward a region of interest, enabling the extraction of information from that area.

![center w:750 h:350](image-21.png)


---
# Background — Delay-and-Sum (DAS)

Delay-and-Sum (DAS) is the simplest linear, digital beamformer: for each image point it computes transmission and reception times of flight, aligns the signals from all transducer elements, and immediately sums them coherently to estimate the amplitude at that point.

---
# Background — Minimum Variance Beamforming (MVB)

Minimum Variance is an adaptive beamformer that computes data-driven weights to minimize output power while maintaining unit response toward the focal point; it suppresses interference and noise, improving resolution and contrast versus non-adaptive methods like DAS.

---
# Background — Ultrafast Imaging

Ultrafast imaging is an ultrasound acquisition mode that illuminates the entire region of interest with plane waves (SPW) and performs parallel receive beamforming, enabling hundreds or thousands of fps instead of traditional line-by-line scanning.

![center w:750 h:350](image-20.png)

---
# Background — Coherent Plane-Wave Compounding (CPWC)

Coherent Plane-Wave Compounding (CPWC) is the natural extension of SPW to recover reconstruction quality. The principle is to transmit multiple SPWs at different steering angles, reconstruct each one, and then sum them.

---
# Problem

In *ultrafast imaging* with SPW, quality (contrast/resolution) is reduced, and CPWC recovers it at the cost of higher computation and lower frame rate.

This defines a *trade-off* between image quality and reconstruction speed.

---
# Objective

Implement and validate a deep learning method to reconstruct from a single SPW with quality comparable to or better than a reconstruction using CPWC.

---
# Solution — Network Architecture

![center w:950 h:450](image-22.png)

---
# Solution — Dataset

- **Source**: Combination of public **PICMUS** and **CUBDL** data to form the training/validation and evaluation sets.

- **Contents**: 561 CPWC acquisitions, all with a 128-element transducer; three angle ranges: 75, 73, and 31 SPWs depending on the acquisition.

- **Model input**: **RF** matrix, image grid, transducer element positions, and parameters (sampling frequency ($f_s$), start time ($t_0$), speed of sound ($c$)).

- **Ground truth**: Reconstruction obtained with MVB using all available angles.

---
# Solution — Pre-processing

- RF matrices have variable size, and we need a fixed input.

- RF input is fixed to 128 × 2800 (128 transducer elements × 2800 samples): if samples are missing → zero padding; if too many → truncation.

- For the output ground truth, a fixed grid of 2048 × 256 is defined.
 
- Normalization follows Sharifzadeh et al. (2023):

  1. Scale by absolute maximum: ( $\mathrm{RF}_{\text{MaxAbs}} = \mathrm{RF}/\max|\mathrm{RF}| \Rightarrow [-1,1]$ ).
  2. Robust step: divide by the standard deviation: ( $\mathrm{RF}_{\text{Robust}} = \mathrm{RF}_{\text{MaxAbs}}/\sigma$ ).

---
# Solution — Training and Evaluation

- **CUBDL** data are split for training/validation at 90%/10%. **PICMUS** acquisitions are used as the test set.

- To compare image quality, the CNR and gCNR contrast metrics are used.

- The network uses MSLE as the training loss and is trained for 50 epochs.

---
# Results — Experimental contrast speckle

![center w:950 h:550](image-23.png)


---

![center w:1100 h:300](image-24.png)

---
# Results — Simulated contrast speckle

![center w:950 h:550](image-25.png)

---

![center w:1100 h:300](image-26.png)

---
# Results — PICMUS carotid longitudinal view

![center w:950 h:550](image-27.png)

---
# Results — PICMUS dataset MSLE table

![center w:900 h:400](image-28.png)

---
# Discussion of results

- **Quality**: early beamforming (Model-1) achieves the most coherent reconstruction with the lowest error.

- **MSLE**: BINN reaches 0.00067 versus 0.00081 for Model-1 in MSLE, but, excluding Field II simulated data, the average drops to 0.00035 with fewer epochs and less training data.

- **Efficiency/risks**: BINN is faster than all models (~430 ms), and overfitting risk persists due to the nature of one **CUBDL** subgroup.

---
# Conclusion

Model-1, which integrates the beamformer layer in the middle of the network, yields reconstructions more faithful to the ground truth, achieves the best MSLE and better contrast than Models 2 and 3, confirming that incorporating beamforming into the architecture is promising, although inference times still do not meet ultrafast imaging standards.

---
# Future work

- **Optimize DAS**: speed up and lighten the DAS layer, currently the bottleneck for time and memory.

- **Expand the dataset**: include more acquisitions with different transducers and targets to reduce overfitting.

- **Different architectures**: evaluate deeper/more modern networks (e.g., ResNet/UNet variants) instead of the simple CNN.
