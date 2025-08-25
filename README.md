### Orange Innovation video3d project

This is the page of Adam SLAM. Here is a list of all published Orange Innovation codes:

* [Adam SLAM - the last mile of camera calibration with 3DGS](../adam-slam)

# Adam SLAM - the last mile of camera calibration with 3DGS
<span class="author-block"><a href="mailto:matthieu.gendrin@orange.com">Matthieu GENDRIN</a>,</span>
<span class="author-block"><a href="mailto:stephane.pateux@orange.com">Stéphane PATEUX</a>,</span>
<span class="author-block"><a href="mailto:xiaoran.jiang@insa-rennes.fr">Xiaoran JIANG</a>,</span>
<span class="author-block"><a href="mailto:theo.ladune@orange.com">Théo LADUNE</a>,</span>
<span class="author-block"><a href="mailto:luce.morin@insa-rennes.fr">Luce MORIN</a></span>

[![button](https://img.shields.io/badge/Paper-blue?style=for-the-badge)](https://hal.science/hal-05131217/document)
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{gendrin2025adam,
  title={Adam SLAM-the last mile of camera calibration with 3DGS},
  author={Gendrin, Matthieu and Pateux, St{\'e}phane and Jiang, Xiaoran and Ladune, Th{\'e}o and Morin, Luce},
  booktitle={ORASIS 2025},
  year={2025}
}</code></pre>
</div>
</section>
The quality of the camera calibration is of major importance for evaluating progresses in novel view synthesis, as
a 1-pixel error on the calibration has a significant impact on the reconstruction quality. While there is no ground
truth for real scenes, the quality of the calibration is assessed by the quality of the novel view synthesis. This paper
proposes to use a [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) model to fine tune calibration by 
backpropagation of novel view color loss with respect to the cameras parameters. The new calibration alone brings 
an average improvement of 0.4 dB PSNR on the dataset used as reference by [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).
The fine tuning may be long and its suitability depends on the criticity of training time, but for calibration
of reference scenes, such as [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), the stake of novel view quality is the most important.

<img src="docs/Adam-SLAM/poster/figures/pc_cams.png">

## How to Install

This project is built on top of the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) and has been tested only on Ubuntu 20.04. If you encounter any issues, please refer to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) for installation instructions.

Clone the Repository, and check-out the appropriate branch:
   ```sh
   git clone --recursive https://github.com/Orange-OpenSource/3dvideo.git
   cd 3dvideo
   git checkout adam-slam
   ```
The rest of the procedure is common with the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

## Fine-tuned calibrations

This project enabled us to fine-tune the calibration of several well-known datasets.
From the test we've led, training with these fine-tuned calibration gives a +0.4dB improvement with
the plain [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) compared to the official calibrations.

| scene | dataset calibration <br> PSNR [dB]  | ours <br> PSNR [dB]    |
| ---   | ---                 | ---       |
| mip360 bicycle | 25.61 | 26.34 | 
| mip360 bonsai | 32.32 | 32.72 | 
| mip360 counter | 29.11 | 29.26 | 
| mip360 garden | 27.77 | 28.32 | 
| mip360 kitchen | 31.55 | 31.14 | 
| mip360 room | 31.72 | 31.94 | 
| mip360 stump | 26.90 | 27.28 | 
| T&T Train | 22.12 | 22.61 | 
| T&T Truck | 25.84 | 26.18 | 
| DB DrJohnson | 29.11 | 30.12 | 
| DB Playroom | 29.96 | 30.77 | 
| <b>Average</b> | <b>28.36</b> | <b>28.79</b> | 

The datasets are available publicly [mip360](https://jonbarron.info/mipnerf360/), [Tanks and Temples](https://www.tanksandtemples.org/), [Deep Blending](http://visual.cs.ucl.ac.uk/pubs/deepblending/datasets.html). Please refer to their conditions of use.

Our fine-tuned calibration are listed in [fine-tuned-calibs](fine-tuned-calibs), they are provided with no guarantee of any sort, but our tests tend to prove that using them as metadata (along with the original images) improve the quality of the reconstruction of +0.4dB in average.

## How to run our code
If you want to fine-tune the calibration of a dataset, you can download, build and run our code.
Running the code is similar to the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) with some differences:
* `--tune_cams` (mandatory) activates the actual fine-tuning of calibration. Beware that this paramter defaults to False, so you have to set it in order to test Adam-SLAM.
* `--tune_from_iter`, `--tune_until_iter` and `--tune_interval` (all optional) determine when the calibration should start and stop in the overal training, and its frequency (see paper for more details).
* At the end of the training, the fine-tuned calibration files will be saved in a `cameras` folder in `model_path` directory.
