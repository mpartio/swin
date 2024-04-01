set -xeu

export PYTHONUNBUFFERED=1

if [ -s train.log ]; then
  mv train.log train.log.$(date +%Y%m%d%H%M%S)
fi

size=475x535
params="effective_cloudiness_heightAboveGround_0 mld_heightAboveGround_0 pres_heightAboveGround_0 fgcorr_heightAboveGround_10 rcorr_heightAboveGround_2 t_heightAboveGround_0 tcorr_heightAboveGround_2 ucorr_heightAboveGround_10 vcorr_heightAboveGround_10 pres_heightAboveSea_0 r_isobaricInhPa_300 t_isobaricInhPa_300 u_isobaricInhPa_300 v_isobaricInhPa_300 z_isobaricInhPa_300 r_isobaricInhPa_500 t_isobaricInhPa_500 u_isobaricInhPa_500 v_isobaricInhPa_500 z_isobaricInhPa_500 r_isobaricInhPa_700 t_isobaricInhPa_700 u_isobaricInhPa_700 v_isobaricInhPa_700 z_isobaricInhPa_700 r_isobaricInhPa_850 t_isobaricInhPa_850 u_isobaricInhPa_850 v_isobaricInhPa_850 z_isobaricInhPa_850 r_isobaricInhPa_925 t_isobaricInhPa_925 u_isobaricInhPa_925 v_isobaricInhPa_925 z_isobaricInhPa_925 r_isobaricInhPa_1000 t_isobaricInhPa_1000 u_isobaricInhPa_1000 v_isobaricInhPa_1000 z_isobaricInhPa_1000"
params=$(echo $params | tr ' ' '\n' | sort | tr '\n' ' ')

date > train.log

python3 pangu_train.py \
	--dataseries_file /data/dataseries/$size/ \
	--parameters $params \
	--batch_size 2 \
	--n_hist 1 \
	--n_workers 2 \
	--device cuda:0 \
	--model_name pangu \
	--input_size $size > train.log 2>&1 &

tail -f train.log
