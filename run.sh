for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python convert_t2i.py $i > output_$i.log 2>&1 &
done