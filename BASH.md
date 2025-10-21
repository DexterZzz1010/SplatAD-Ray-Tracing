```
sbatch --partition=zprod scripts/train_splatad.sh 
```

sbatch --partition=ztestpreemp scripts/render.sh /staging/fisheye/mthesis/splatad/code/84778386_1358268_utQeIs/outputs/unnamed/splatad/2025-10-01_165531/nerfstudio_models/step-000030000.ckpt
sbatch --partition=ztestpreemp scripts/render.sh /staging/fisheye/mthesis/splatad/runs/10-03_11-09_2025_splatad_nuscenes_84778386/reproduce-adsplat-config/splatad/2025-10-03_110929/nerfstudio_models/step-000030000.ckpt

sbatch --partition=ztestpreemp scripts/render.sh /staging/fisheye/mthesis/splatad/runs/10-03_17-57_2025_splatad_nuscenes_84778386/nuscenes-config1-optimized/splatad/2025-10-03_175809/nerfstudio_models/step-000030000.ckpt


sbatch --partition=ztestpreemp scripts/train_splatgut.sh 


tensorboard --logdir shares/staging/fisheye/mthesis/splatad/runs/


