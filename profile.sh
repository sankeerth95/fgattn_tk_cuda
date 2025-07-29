

# nsys -o nsysreport.nsys-rep --trace=cuda ./a.out

ncu -o ncureport.ncu-rep -f --set=full --kernel-name attn_fwd_kernel  --launch-skip 5 --launch-count 1 "./a.out"

