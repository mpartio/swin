set -x

export PYTHONUNBUFFERED=1

if [ -z "$DATA_PATH" ]; then
  export DATA_PATH=$HOME/cloudnwc/effective_cloudiness/data/dataseries/npz/224x224
fi

if [ -z "$BATCH_SIZE" ]; then
  export BATCH_SIZE=16
fi

if [ -z "$MODEL_NAME" ]; then
  export MODEL_NAME=swin-large-patch4-window7
fi

set -ue

if [ -s train.log ]; then
  mv train.log train.log.$(date +%Y%m%d%H%M%S)
fi

date > train.log

python3 train.py \
	--leadtime_conditioning 12 \
	--n_hist 4 \
	--n_pred 1 \
	--model_name $MODEL_NAME \
	--dataseries_file $DATA_PATH/nwcsaf-effective-cloudiness-20190801-20200801-img_size=224x224-float32.npz \
	--model_dir /tmp/$MODEL_NAME \
	--batch_size $BATCH_SIZE 2>&1 | ts | tee -a train.log

date >> train.log

cp train.log /tmp/$MODEL_NAME
