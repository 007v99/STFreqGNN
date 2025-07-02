name='ABIDE_onlyGFRN_Pearson_3frq_5e-5lr_1head_b128_e75_mask'

batch_size=128
datapath='ABIDE'
splitpath='split_ABIDE'
reprocess=0 #Whether to force re-preprocessing of data

#Training strategy
mod='train'
#mod='analysize'
lr=5e-5
warmUp=0
epochs=10

#model config
input_encoder='GFRN'
#input_encoder='MLP'
cls=0
mask=1

#node_feature=TimeSeries
node_feature="one-hot"
#node_feature="LaplacianEigenvectors"
#node_feature="Pearson"
inputShape='90,6,3,90'
transformer_input_dim=128
drop_ratio=0.3
transLayer=4
encoderLayer=1
attHeads=1

gpulist=(1 2 3 4 5)
foldlist=(1 2 3 4 5)

for i in {0..0}
do
{
folds=${foldlist[i]}
gpu=${gpulist[i]}
python FTD_train.py \
    --name $name \
    --batch_size $batch_size \
    --lr $lr \
    --warmUp $warmUp \
    --folds $folds \
    --epochs $epochs \
    --mod $mod \
    --gpu $gpu \
    --datapath $datapath \
    --splitpath $splitpath \
    --cls $cls \
    --reprocess $reprocess \
    --node_feature $node_feature \
    --input_encoder $input_encoder \
    --inputShape $inputShape \
    --transLayer $transLayer \
    --encoderLayer $encoderLayer \
    --attHeads $attHeads \
    --transformer_input_dim $transformer_input_dim \
    --drop_ratio $drop_ratio \
    --mask $mask
}&
done
wait
echo "Finish!"