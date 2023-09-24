# How to use
## Project structure
Generally, the structure of the pipeline is separated into distinct **projects** for different resource packs:
  - Project_1 (Resource Pack A)
    - Upscale_date/time
      - ...
    - Upscale_date/time
      - ...
    - ...
  - Project_2 (Resource Pack B)
    - Upscale_date/time
      - ...
    - ...
  - ...
  - Project_N (Resource Pack N)
    - ...

Each "Upscale_date/time" folder represents training for a given hyperparameters configuration


  - Upscale_date/time
      - hyperparameters.yml
      - train_loop_{epoch}.pth.tar
      - train_loop_{epoch} *empirical testing*
        - texture_comparison_1.pdf
        - texture_comparison_2.pdf
        - ...
    - ...
  - ...


## Starting a project *(for given texture pack)*
### Setup for a particular dataset (for particular resource pack):
0) To begin, initialize `config.yml`:
   1) Set `project_dir` to the folder you would like to use for your project
   2) Set `scaling_factor` to 2 for 32x pack, 4 for 64x pack
   3) Set `n_blocks` to 8 for 32x pack, 10 for 64x pack
   4) Set `device` to whatever hardware you have available
1) First, you must prepare the data:
   1) Run `python run.py prepare [feature_pack_dir] [label_pack_dir]`
   2) Where `feature_pack_dir` is the location of the default minecraft textures
   3) Where `label_pack_dir` is the location of the custom textures - the model will learn to upscale the default textures to the custom ones

You now have your data prepared, and can proceed with training

## Begin training
Now we can begin training our model for a given hyperparameters configuration
0) To begin, set values in `hyperparameters.yml` - for starting out, it is recommended to just use default values
1) To make further adjustments, modify `train.py`
2) Finally, to start training, run `python run.py train`

One last note: a framework for using other models (besides ResNetImplementation) in this pipeline will come in the future, but it is still a work in progress. Here is how it will work:

Update `from TextureGAN.Model.Upscale.[ModelName].PerformTraining import PerformTraining` 
with whatever model you are using. 

Ensure that config, hyperparameters of each model must correspond to `config.yml`, `hyperparameters.yml`
## Begin testing *(empirical testing)*
I will preface this by noting that adding quantitative test metrics is still a work in progress. For now, we can empirically test by feeding textures from a mod into our model and seeing whether we like the result.
0) To begin, run `python run.py testmod [jar_file] [run_name]`
