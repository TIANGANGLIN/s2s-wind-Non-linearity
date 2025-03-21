#!/bin/bash
clear
# Current
# echo '{
#     "url"   : "https://api.ecmwf.int/v1",
#     "key"   : "27391a591f30dc67c0db42964bba26d9",
#     "email" : "ganglin.tian@lmd.ipsl.fr"
# }' > $HOME/.ecmwfapirc

# echo '{
#     "url"   : "https://api.ecmwf.int/v1",
#     "key"   : "837621f4d75eab6313f93c8c47077738",
#     "email" : "ganglin.tian@imt-atlantique.net"
# }' > $HOME/.ecmwfapirc

# echo 'url: https://cds.climate.copernicus.eu/api
# key: e9d2cc21-113e-42e5-b887-15209904d115' > $HOME/.cdsapirc

# cd /net/nfs/ssd1/tganglin/toGit
# conda activate /net/nfs/ssd1/tganglin/env/nonlinear
# python src/downscaling/data/downloader_ensembles.py
# python src/downscaling/data/downloader_reanalysis.py

# for trainingModels in MLR CNN
for trainingModels in CNN MLR
do 
    python -u src/scripts/template_main.py \
    --debug 0 \
    --trainingModels ${trainingModels} \
    --trainDailyMean 0 \
    --predors z500 \
    --prednds ws100 \
    --useOptuna 1

    # python -u src/scripts/template_main.py \
    # --debug 0 \
    # --trainingModels ${trainingModels} \
    # --trainDailyMean 0 \
    # --predors z500 \
    # --prednds 10uv \
    # --useOptuna 1
done

# python src/scripts/evaluation/evalute.py --fold 0 --dir data/results/WithOptuna
# python src/scripts/evaluation/evalute.py --fold 1 --dir data/results/WithOptuna
# python src/scripts/evaluation/evalute.py --fold 2 --dir data/results/WithOptuna