python build_embeddings.py \
    /srv/nfs/VESO/fedor/labs/prog2/data/videos \
    /srv/nfs/VESO/fedor/labs/prog2/data/aggl.csv \
    --model_path /srv/nfs/VESO/fedor/labs/prog2/models/videomae-large-finetuned-kinetics \
    --cuda 1 \
    --aggregate