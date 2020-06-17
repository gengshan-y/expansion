modelname=$1
filename=$modelname-$2
array=(
#1999 3999 5999 7999 
9999 11999 13999 15999 17999 19999 21999 23999 25999 27999 29999
# 31999 33999 35999 37999 39999 41999 43999 45999 47999 49999 51999 53999 55999 57999 59999
# 61999 63999 65999 67999 69999 71999 73999 75999 75999 77999 79999 81999 83999 85999 87999 89999 91999 93999 95999 97999 99999 101999 103999 105999 107999 109999 111999 113999 115999 117999 119999 121999 123999 125999 127999 129999 131999 133999 135999 137999 139999 141999 143999 145999 147999 149999 151999 153999 155999 157999 159999 161999 163999 165999 167999 169999 171999 173999 175999 177999 179999 181999  183999 185999 187999 189999 191999 193999 195999 197999 199999 201999 203999 205999 207999 209999 211999 213999 215999 217999 219999 221999 223999 225999 227999 229999 231999 233999 235999 237999 239999 241999 243999 245999 247999 249999 251999 253999 255999 257999 259999 261999 263999 265999 267999 269999
)

for i in "${array[@]}"
do
  ## Sintel val
  #echo $i >> results/os-$filename
  #CUDA_VISIBLE_DEVICES=0 python submission.py --dataset sintel --datapath /ssd/rob_flow/training/   --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.pth  --testres 1
  #python eval_exp.py   --path ./weights/$modelname/  --dataset sintel >> results/os-$filename
  #python eval_flow.py --path ./weights/$modelname/ --vis no --dataset sintel >> results/os-$filename

  # KITTI val
  echo $i >> results/ok-$filename
  #CUDA_VISIBLE_DEVICES=1 python submission.py --dataset 2015val --datapath /ssd/kitti_scene/training/   --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.pth  --testres 1
  CUDA_VISIBLE_DEVICES=1 python submission.py --dataset 2015val --datapath /ssd/kitti_scene/training/   --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.pth  --testres 1 --fac 2 --maxdisp 512
  python eval_exp.py   --path ./weights/$modelname/  --dataset 2015val >> results/ok-$filename
  python eval_flow.py       --path ./weights/$modelname/ --vis no --dataset 2015val >> results/ok-$filename
done
