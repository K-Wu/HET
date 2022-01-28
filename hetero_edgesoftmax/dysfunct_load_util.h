#pragma once
#include <stdio.h>
#include <stdlib.h>

long get_file_size(const char* filename){
    FILE * pFile = fopen(filename, "rb");
    if (!pFile) {
        printf("get_file_size open file fails %s %d\n", filename, errno);
        exit(errno);
    }
    fseek (pFile , 0 , SEEK_END);
    long lSize = ftell (pFile);
    //rewind (pFile);
    fclose (pFile);
    return lSize;
}

int* _load(const char* filename, size_t num_int_in_file, int& max_idx){
    int* data_ptr = (int*)malloc((long long)num_int_in_file * sizeof(int));
    FILE* fin = fopen(filename, "rb");
    if (!fin) {
        printf("_load open file fails %s %d\n", filename,errno);
        exit(errno);
    }
    fread(data_ptr, sizeof(int), num_int_in_file, fin);
    fclose(fin);
    max_idx = -1;
    for(int idx = 0; idx<num_int_in_file; idx++){
        if (data_ptr[idx] > max_idx)
            max_idx = data_ptr[idx];
    }
    return data_ptr;
}


int test(){
    int written_by_max_idx, has_max_idx, is_about_max_idx, cited_max_idx, citing_max_idx, writing_max_idx;
    long written_by_num_int_in_file = get_file_size("data/written-by_coo_1.npy")/sizeof(int);
    long has_num_int_in_file = get_file_size("data/has_coo_1.npy")/sizeof(int);
    long is_about_num_int_in_file = get_file_size("data/is-about_coo_1.npy")/sizeof(int);
    long cited_num_int_in_file = get_file_size("data/cited_coo_1.npy")/sizeof(int);
    long citing_num_int_in_file = get_file_size("data/citing_coo_1.npy")/sizeof(int);
    long writing_num_int_in_file = get_file_size("data/writing_coo_1.npy")/sizeof(int);
    int* written_by_data_ptr = _load("data/written-by_coo_1.npy", written_by_num_int_in_file, written_by_max_idx);
    int* has_data_ptr = _load("data/has_coo_1.npy", has_num_int_in_file, has_max_idx);
    int* is_about_data_ptr = _load("data/is-about_coo_1.npy", is_about_num_int_in_file, is_about_max_idx);
    int* cited_data_ptr = _load("data/cited_coo_1.npy", cited_num_int_in_file, cited_max_idx);
    int* citing_data_ptr = _load("data/citing_coo_1.npy", citing_num_int_in_file, citing_max_idx);
    int* writing_data_ptr = _load("data/writing_coo_1.npy", writing_num_int_in_file, writing_max_idx);
}