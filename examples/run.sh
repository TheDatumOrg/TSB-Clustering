#!/bin/bash

cd ../TSClusterX/

for experiment in {1..10}; do
    echo "Running experiment $experiment"
    
    # partitional clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kmeans --clust_class partitional --distance euclidean --parameter_settings parameters/kmeans.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kdba --clust_class partitional --parameter_settings parameters/kdba.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model ksc --clust_class partitional --parameter_settings parameters/ksc.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kshape --clust_class partitional --parameter_settings parameters/kshape.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kasba --clust_class partitional --parameter_settings parameters/kasba.json --metrics RI ARI NMI --experiment $experiment

    # partitioning around medoids:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance euclidean --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance msm --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance twed --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance erp --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance sbd --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance swale --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance dtw --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance edr --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model pam --clust_class partitional --distance lcss --parameter_settings parameters/pam.json --metrics RI ARI NMI --experiment $experiment

    # kernel-based clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kernelkavg --clust_class kernel --distance sink --parameter_settings parameters/kernelkavg.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kernelkavg --clust_class kernel --distance gak --parameter_settings parameters/kernelkavg.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kernelkavg --clust_class kernel --distance kdtw --parameter_settings parameters/kernelkavg.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model kernelkavg --clust_class kernel --distance rbf --parameter_settings parameters/kernelkavg.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model spectralclustering --clust_class kernel --distance sink --parameter_settings parameters/spectralclustering.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model spectralclustering --clust_class kernel --distance gak --parameter_settings parameters/spectralclustering.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model spectralclustering --clust_class kernel --distance kdtw --parameter_settings parameters/spectralclustering.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model spectralclustering --clust_class kernel --distance rbf --parameter_settings parameters/spectralclustering.json --metrics RI ARI NMI --experiment $experiment

    # hierarchical clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model birch --clust_class hierarchical --parameter_settings parameters/birch.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model agglomerative --clust_class hierarchical --distance euclidean --parameter_settings parameters/agglomerative.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model agglomerative --clust_class hierarchical --distance sbd --parameter_settings parameters/agglomerative.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model agglomerative --clust_class hierarchical --distance msm --parameter_settings parameters/agglomerative.json --metrics RI ARI NMI --experiment $experiment

    # distribution-based clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model gaussianmixture --clust_class distribution --parameter_settings parameters/gaussianmixture.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model affinitypropagation --clust_class distribution --distance euclidean --parameter_settings parameters/affinitypropagation.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model affinitypropagation --clust_class distribution --distance sbd --parameter_settings parameters/affinitypropagation.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model affinitypropagation --clust_class distribution --distance msm --parameter_settings parameters/affinitypropagation.json --metrics RI ARI NMI --experiment $experiment

    # density-based clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model densitypeaks --clust_class density --distance euclidean --parameter_settings parameters/densitypeaks.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model densitypeaks --clust_class density --distance sbd --parameter_settings parameters/densitypeaks.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model densitypeaks --clust_class density --distance msm --parameter_settings parameters/densitypeaks.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model dbscan --clust_class density --distance euclidean --parameter_settings parameters/dbscan.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model dbscan --clust_class density --distance sbd --parameter_settings parameters/dbscan.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model dbscan --clust_class density --distance msm --parameter_settings parameters/dbscan.json --metrics RI ARI NMI --experiment $experiment

    # shapelet-based clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model ushapelets --clust_class shapelet --parameter_settings parameters/ushapelets.json --metrics RI ARI NMI --experiment $experiment

    # model-based clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model ar_coeff --clust_class model --parameter_settings parameters/ar_coeff.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model catch22 --clust_class model --parameter_settings parameters/catch22.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model es_coeff --clust_class model --parameter_settings parameters/es_coeff.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model ar_pval --clust_class model --parameter_settings parameters/ar_pval.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model lpcc --clust_class model --parameter_settings parameters/lpcc.json --metrics RI ARI NMI --experiment $experiment

    # deep learning-based clustering:
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model dec --clust_class deep --parameter_settings parameters/dec.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model ofa --clust_class deep --parameter_settings parameters/ofa.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model chronos --clust_class deep --parameter_settings parameters/chronos.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model moment --clust_class deep --parameter_settings parameters/moment.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model idec --clust_class deep --parameter_settings parameters/idec.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model depict --clust_class deep --parameter_settings parameters/depict.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model sdcn --clust_class deep --parameter_settings parameters/sdcn.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model dtc --clust_class deep --parameter_settings parameters/dtc.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model vade --clust_class deep --parameter_settings parameters/vade.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model dcn --clust_class deep --parameter_settings parameters/dcn.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model som_vae --clust_class deep --parameter_settings parameters/som_vae.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model dtcr --clust_class deep --parameter_settings parameters/dtcr.json --metrics RI ARI NMI --experiment $experiment
    python main.py --dataset ucr_uea --start 1 --end 128 --dataset_path ../data/UCR2018/ --model clustergan --clust_class deep --parameter_settings parameters/clustergan.json --metrics RI ARI NMI --experiment $experiment
    
    echo "Completed experiment $experiment"
done

echo "All experiments completed!"