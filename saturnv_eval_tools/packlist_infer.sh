#!/bin/bash
export PYTHONPATH=${WORKING_PATH}:${PYTHONPATH}

source /opt/miniconda3/etc/profile.d/conda.sh

conda activate easyvolcap
export CC=/usr/local/gcc-11.4/bin/gcc
export CXX=/usr/local/gcc-11.4/bin/g++
which python3
python3 -m pip install numexpr==2.11.0 pillow==11.2.1 evo==1.31.1 pykdtree==1.4.3 numba==0.61.2 gtsam horizon_driving_dataset numba  -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

cd ${WORKING_PATH}/saturnv_eval_tools/pgo
rm -r build
mkdir build
cd build
cmake ..
make -j

cd ${WORKING_PATH}

save_root=$1
weight_path=$2
config_path=$3
weight_path_lc=$4
config_path_lc=$5
pandar_gtpath=$6
site_rootpath=$7
site=$8
gt_type=$9
mvseq=${10}
use_cam_emb=${11}
save_pointcloud=${12}
sensors=${13}
add_loop_closure=${14}

echo "Running packlist_infer with:"
echo "  save_root      = $save_root"
echo "  weight_path    = $weight_path"
echo "  config_path    = $config_path"
echo "  weight_path_lc = $weight_path_lc"
echo "  config_path_lc = $config_path_lc"
echo "  pandar_gtpath  = $pandar_gtpath"
echo "  site_rootpath  = $site_rootpath"
echo "  site           = $site"
echo "  gt_type        = $gt_type"
echo "  mvseq          = $mvseq"
echo "  use_cam_emb    = $use_cam_emb"
echo "  save_pointcloud= $save_pointcloud"
echo "  sensors        = $sensors"
echo "  add_loop_closure= $add_loop_closure"



echo "Running packlist_infer"

which python3

python3 ${WORKING_PATH}/saturnv_eval_tools/packlist_infer.py \
  --save_root=$save_root \
  --weight_path=$weight_path \
  --config_path=$config_path \
  --weight_path_lc=$weight_path_lc \
  --config_path_lc=$config_path_lc \
  --pandar_gtpath=$pandar_gtpath \
  --site_rootpath=$site_rootpath \
  --site=$site \
  --gt_type=$gt_type \
  --mvseq=$mvseq \
  --use_cam_emb=$use_cam_emb \
  --save_pointcloud=$save_pointcloud \
  --sensors=$sensors \
  --add_loop_closure=$add_loop_closure
