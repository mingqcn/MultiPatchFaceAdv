PYTHONPATH=":/data/hd/face-adv/Attack"
export PYTHONPATH
python3 adv_patch.py --input_path=data/aligned \
                     --output_path=data/kmeans_output \
                     --eps=2 \
                     --min_samples=10 \
                     --model=models/r100.pb \
                    --max_percent=0.2 \
                    --min_percent=0.01 \
                    --gradient_step=51 \
                    --max_cluster=50 \
                    --cluster_method=kmeans \
                    --size=1