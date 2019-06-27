
is_opt=''
for run_mode in '' '_bayes'
do
    for model in 'lstm' 'phoible' 'phoible-lookup'
    do
        echo ''
        echo ''
        echo $model
        echo ''
        python learn_layer/train_base${run_mode}.py --model $model $is_opt

        echo ''
        echo ''
        echo 'shared-'${model}
        echo ''
        python learn_layer/train_shared${run_mode}.py --model shared-${model} $is_opt
    done
done

is_opt='--opt'
run_mode='_cv'
for model in 'lstm' 'phoible' 'phoible-lookup'
do
    echo ''
    echo ''
    echo $model
    echo ''
    python learn_layer/train_base${run_mode}.py --model $model $is_opt

    echo ''
    echo ''
    echo 'shared-'${model}
    echo ''
    python learn_layer/train_shared${run_mode}.py --model shared-${model} $is_opt
done

model='ngram'
for run_mode in '' '_cv'
do
    echo ''
    echo ''
    echo $model
    echo ''
    python learn_layer/train_ngram${run_mode}.py --model $model
done

model='unigram'
for run_mode in '' '_cv'
do
    echo ''
    echo ''
    echo $model
    echo ''
    python learn_layer/train_unigram${run_mode}.py --model $model
done

for run_mode in '' '_bayes' '_cv'
do
    for artificial_type in 'devoicing' 'harmony'
    do
        python learn_layer/train_artificial${run_mode}.py --artificial-type ${artificial_type}
    done
done

for artificial_type in 'devoicing' 'harmony'
do
    python learn_layer/train_artificial_ngram.py --artificial-type ${artificial_type} --model ngram
done
