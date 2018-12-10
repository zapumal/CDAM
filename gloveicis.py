from __future__ import print_function
import os
import numpy as np
import falconn
import timeit
import math
import gc

#the code is based on https://github.com/FALCONN-LIB/FALCONN library for LSH

#lda representation of the two datasets
f1 = 'dataseticis/lsh_stack_500_max'
f2 = 'dataseticis/lsh_link_500_max'

if __name__ == '__main__':
    dataset_file = f1 + '.npy'
    # we build only 100 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number
    number_of_tables = 100
    number_of_probes = number_of_tables 

    print('Reading the dataset')
    dataset = np.load(dataset_file)
    print('Done')

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    assert dataset.dtype == np.float32

    # Normalize all the lenghts, since we care about the cosine similarity.
    print('Normalizing the dataset')
    #dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    print('Done')

    #read the file 2 to match
    dataset_file2 = f2 + '.npy'

    print('Reading the dataset2')
    dataset2 = np.load(dataset_file2)
    print('Done')

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    assert dataset2.dtype == np.float32

    l_d = len(dataset)
    l_d2 = len(dataset2)

    link_ids = dict()
    stack_ids = dict()

    idcounter = 0
    #load dataset 1 to map
    with open(f2 + ".out") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.split("\t")
            link_ids[idcounter] = line[0]
            idcounter += 1

    idcounter = 0
    #load dataset 2 to map
    with open(f1 + ".out") as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            line = line.split("\t")
            stack_ids[idcounter] = line[0]
            idcounter += 1

    # Normalize all the lenghts, since we care about the cosine similarity.
    print('Normalizing the dataset2')
    dataset2 /= np.linalg.norm(dataset2, axis=1).reshape(-1, 1)
    print('Done')

    # Center the dataset and the queries: this improves the performance of LSH quite a bit.
    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    center = np.mean(dataset2, axis=0)
    dataset2 -= center
    print('Done')

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    #params_cp.lsh_family = falconn.LSHFamily.Hyperplane 
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    ##params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
    params_cp.l = number_of_tables
    # we set one rotation, since the data is dense enough,
    # for sparse data set it to 2
    params_cp.num_rotations = 2
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 0
    #params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    params_cp.storage_hash_table = falconn.StorageHashTable.LinearProbingHashTable
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(16, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format(t2 - t1))


    #probing operation start
    done_users = list()
    map_answers = dict()
    imatch = 0
    total_answers = 0

    map_candidates = dict()

    #init the map
    for user in range(0, len(dataset2)):
        usid = link_ids[user]
        map_candidates[usid] = dict()
    
    while True: 
        print('xxxx Choosing number of probes', number_of_probes)
        total_answers_wth_unmatch = 0

        query_object = table.construct_query_object()
        query_object.set_num_probes(number_of_probes)
        
        for user in range(0, len(dataset2)): 
            
            if link_ids[user] in done_users:
                continue
                
            query = dataset2[user]  
            answers = query_object.get_unique_candidates(query)
            total_answers_wth_unmatch += len(answers)

            tmpmap = map_candidates[link_ids[user]]        
        
            for ij in answers:

                tmpmap[stack_ids[ij]] = stack_ids[ij]

                if link_ids[user] == stack_ids[ij]:
                    imatch += 1
                    ####print(user, imatch, len(answers), link_ids[user], stack_ids[ij])

                    total_answers += len(answers)
                    total_answers_wth_unmatch -= len(answers)

                    done_users.append(link_ids[user])
                    #map_answers[link_ids[user]] = answers

            map_candidates[link_ids[user]] = tmpmap

        total_answers_wth_unmatch += total_answers 
        a1 = (l_d*1.0 - ((total_answers * 1.0) / imatch))/(l_d*1.0)
        a2 = (l_d*1.0 - ((total_answers_wth_unmatch * 1.0)/l_d2))/(l_d*1.0)
        a3 = (imatch*1.0)/(l_d2*1.0)

        #calculate total candidates
        newtotalcount = 0
        for user in range(0, len(dataset2)):
            usid = link_ids[user]
            tmpmap = map_candidates[usid]
            newtotalcount += len(tmpmap)

        a4 = (1.0 - ((newtotalcount* 1.0)/(l_d2 * l_d)))

        print("xxxx reducton in search space:", a1) 
        print("xxxx reducton in search space(all):", a2) 
        print("xxxx reducton in search space(new):", a4) 
        print("xxxx success users out of ", l_d2, ":", imatch, a3) 
        print("++++", a1, a2, a3, a4) 

        if a3 > 0.95:
            break

        number_of_probes = number_of_probes + number_of_tables
