#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

//同时初始化page_table和swap_table
__device__ void init_invert_page_table(VirtualMemory *vm) {
  //初始化page table
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1 (invalid指值是空的)
    //   valid bit + pid + pagenum + lrutimer  = 32bit
    //     1bit      2bit    13bit     16bit

    //vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;  //不知道什么意思？
  }

  //初始化swap table
  for (int i = 0; i < 4096; i++) {
    vm->swap_table[i] = 0x80000000; // invalid := MSB is 1 (invalid指值是空的)
    //   valid bit + pid + pagenum   = 32bit
    //     1bit      2bit    29bit     

  }
}

//在page table里面找vpn在对应的physical memory (data buffer) 的frame number位置
__device__ int search_page_table(VirtualMemory *vm, int vpn){
  u32 page_sequence;
  for(int i = 0; i < 1024; i++){
    page_sequence = vm->invert_page_table[i];   //那个32位数，里面包含了page number
    if( ((page_sequence >> 31) == 0) && ( (page_sequence & 0x1FFF0000) == (u32)vpn<<16 )  ){   //该frame index对应的vpn值非空，且找到匹配
      return i;
    }
  }
  return -1; //没找到vpn对应的frame index值
}

//在swap table里面找vpn对应的disk的index位置
__device__ int search_swap_table(VirtualMemory *vm, int vpn){
  u32 page_sequence;
  for(int i = 0; i < 4096; i++){
    page_sequence = vm->swap_table[i];
    if( ((page_sequence >> 31) == 0) && ( (page_sequence & 0x1FFFFFFF) == vpn )  ){   //该disk index对应的vpn值非空，且找到匹配
      return i;
    }
  }
  return -1; //没找到vpn对应的disk index值
}

//swap的时候，要互换disk和memory里面的内容
__device__ void exchange_mem_disk_content(VirtualMemory *vm, int frame_num, int disk_num){
  u32 temp;
  for(int i = 0; i<32; i++){   //这个page里面全部的内容（32byte，1 byte 1 行）都要换
    temp = vm->buffer[frame_num * 32 + i];
    vm->buffer[frame_num * 32 + i] = vm->storage[disk_num * 32 + i];
    vm->storage[disk_num * 32 + i] = temp;
  }
}

//找invert page table里有没空位,找到第一个空位即可
__device__ int find_blank_page_table(VirtualMemory *vm){
  int blank_index = -1;
  for(int i = 0; i < 1024; i++){      //找page table中有没空位,找到第一个空位即可
    if( ( (vm->invert_page_table[i])>>31 ) ){ // 判断第一位是不是1(invalid,即空)
      blank_index = i; 
      break;
    }
  }
  return blank_index;
}

//增加page table中所有人（不管里面有没有存pt）的timer值
__device__ void increase_timer(VirtualMemory *vm){
  for(int i = 0; i < 1024; i++){
    (vm->invert_page_table[i])++;  // 不管里面没有存东西，LRU_timer都加1
        //page = ((vm->invert_page_table[i] & 0x1FFF0000) >> 16);
        //printf("page table: frame %d stores page %d\n", i, page);
  }
}

//找出最后一次被访问时间距离现在最久的page（后16位LRU timer最大的page）
__device__ int find_swap_index(VirtualMemory *vm){
  int LRU_index, LRU_time;   //page table中应该被swap出来的那个内存块的index  
  int max_time = -1; 
  for(int i = 0; i < 1024; i++){
    if( ( (vm->invert_page_table[i]) >> 31 ) == 0 ){  //这个检验或许多余？毕竟默认是满的 
      LRU_time = (vm->invert_page_table[i] & 0x0000FFFF);  //获取这个page sequence的LRU timer值（后16位）
      if(LRU_time > max_time){
        max_time = LRU_time;
        LRU_index = i;
      }
    }
  }
  return LRU_index;
}

// vm_init(&vm, data, storage, pt, st, &pagefault_num, PAGE_SIZE,
//           INVERT_PAGE_TABLE_SIZE, SWAP_TABLE_SIZE, PHYSICAL_MEM_SIZE, STORAGE_SIZE,
//           PHYSICAL_MEM_SIZE / PAGE_SIZE);
// 所以PAGE_ENTRIES = PHYSICAL_MEM_SIZE / PAGE_SIZE = 1024
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, u32 *swap_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE, int SWAP_TABLE_SIZE, 
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->swap_table = swap_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->SWAP_TABLE_SIZE = SWAP_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  __syncthreads();
  if (addr % 4 == threadIdx.x){

    int disk_index;
    uchar value;
    u32 vpn = addr >> 5;
    u32 offset = addr & 0b11111;
    //int page;
    int frame_num = search_page_table(vm, vpn);
    int blank_index = -1;
    if (frame_num != -1){                    // page在内存中
      increase_timer(vm);     //记得增加所有人的timer
      vm->invert_page_table[frame_num] &= 0xFFFF0000; 
      (vm->invert_page_table[frame_num])++;   // 把刚刚找到的page的历史设为 1
      value = vm->buffer[frame_num * 32 + offset];  //直接找到frame_num对应位置然后把数读出来
      // return value;
    }
    else{                                   // page不在内存中
      *(vm->pagefault_num_ptr) += 1;
      //printf("read page fault: page number %d\n", vpn);
      increase_timer(vm);     //记得增加所有人的timer

      blank_index = find_blank_page_table(vm); //找page table中有没空位,找到第一个空位即可

      if (blank_index != -1){       //page table（即physical memory）有空位，可直接换过来
        // int disk_index;
        disk_index = search_swap_table(vm, vpn);    //找到这个page number在disk中的位置（肯定有的，不然题没法做了）
        exchange_mem_disk_content(vm, blank_index, disk_index);   //swap，把memory这个空block换到disk去，把disk的block换过来
        vm->invert_page_table[blank_index] &= 0x00000000;   
        vm->invert_page_table[blank_index] |= (u32)vpn<<16;  //把目标vpn存入memory这个位置的page sequence中
        (vm->invert_page_table[blank_index])++;      //page sequence的timer bit设为1
        vm->swap_table[disk_index] &= 0x00000000;    //因为swap回来了，所以要把swap table对应位置的值删掉，初始化为0x80000000
        vm->swap_table[disk_index] |= 0x80000000;  
        value = vm->buffer[blank_index * 32 + offset];  //从memory对应位置读数出来
        //return value;
      }
      //if (blank_index == -1){
      else{               //page table没空位，那要swap
        disk_index = search_swap_table(vm, vpn);  //找出这个pagenumber在disk中的位置

        int LRU_index = find_swap_index(vm);  //找出最后一次被访问时间距离现在最久的page（后16位LRU timer最大的page）

        int out_page_num;   //physical memory中应该被swap出去的page number
        out_page_num = (vm->invert_page_table[LRU_index]) & 0x1FFF0000;
        out_page_num = out_page_num >> 16;  //获取要从memory中被swap出去的那个page number值

        exchange_mem_disk_content(vm, LRU_index, disk_index);   // 交换memory和disk中的block的值
        vm->invert_page_table[LRU_index] &= 0x00000000;    // 修改page table
        vm->invert_page_table[LRU_index] |= (u32)vpn<<16; 
        (vm->invert_page_table[LRU_index])++;  
        vm->swap_table[disk_index] &= 0x00000000;    
        //vm->swap_table[disk_index] |= 0x80000000;  //修改swap table,初始化swap出来那个位置的值
        vm->swap_table[disk_index] |= out_page_num;   //修改swap table，对应位置值变成swap出去的那个page
        value = vm->buffer[LRU_index * 32 + offset];   // 修改内存中的对应地址值
      }
    }  
    return value; //TODO
  }
  return;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  __syncthreads();
  if (addr % 4 == threadIdx.x){

    u32 vpn = addr >> 5;    //virtual page number
    u32 offset = addr & 0b11111;
    int disk_index = -1;
    int frame_num = search_page_table(vm, vpn);
    int blank_index = -1;

    if (frame_num != -1){                   //vpn在page table中
      vm->buffer[frame_num * 32 + offset] = value;   
      increase_timer(vm);     //记得增加所有人的timer
      vm->invert_page_table[frame_num] &= 0xFFFF0000; // 把刚刚操作的page的历史清空成 0
      vm->invert_page_table[frame_num] &= 0xFFFFFFF1;;   // 把刚刚操作的page的历史设为 1
    }

    else{                     //frame number不在page table里面
      // printf("page fault: page number %d", vpn)
      (*(vm->pagefault_num_ptr)) += 1;
      increase_timer(vm);     //记得增加所有人的timer

      blank_index = find_blank_page_table(vm); //找page table中有没空位,找到第一个空位即可

      if (blank_index != -1){       //page table（即physical memory）有空位，可直接写入
        vm->invert_page_table[blank_index] &= 0x00000000;  //把valid设为0，即该位置将被填入数字
        vm->invert_page_table[blank_index] |= ( (u32)vpn << 16 );  // 把29～17位改成vpn
        (vm->invert_page_table[blank_index])++;     //把这个新的空位历史值设为1
        vm->buffer[blank_index * 32 + offset] = value;   //对应物理地址值写入value
      }

      else{                         //page table (physical memory)没空位了,需要swap
        disk_index = search_swap_table(vm, vpn);  // 试图找出这个pagenumber在disk中的位置

        int LRU_index = find_swap_index(vm);  //找出最后一次被访问时间距离现在最久的page（后16位LRU timer最大的page）

        int out_page_num;   //physical memory中应该被swap出去的page number
        out_page_num = ((vm->invert_page_table[LRU_index]) & 0x1FFF0000);
        out_page_num = out_page_num >> 16;  //获取要从memory中被swap出去的那个page number值

        if(disk_index == -1){           //disk里也没找到(一般这种情况会在第一次写入数据的时候出现)
          int disk_empty_index = -1;    //在swap table里面找出disk中第一个空位（一定会找到，因为默认disk不写满）

          for(int i = 0; i < 4096; i++){      //找swap table (即disk）中第一个空位
            if( ( (vm->swap_table[i])>>31 ) ){ // 判断第一位是不是1(invalid,即空)
              disk_empty_index = i; 
              break;
            }
          }  
          exchange_mem_disk_content(vm, LRU_index, disk_empty_index);   //swap，把physical memory中LRU那个block换到disk去，把disk的空内存block换过来，以供写入
          vm->invert_page_table[LRU_index] &= 0x00000000;   
          vm->invert_page_table[LRU_index] |= (u32)vpn<<16;  //把目标vpn存入memory这个位置的page sequence中
          (vm->invert_page_table[LRU_index])++;      //page sequence的timer bit设为1
          vm->swap_table[disk_empty_index] &= 0x00000000;    
          vm->swap_table[disk_empty_index] |= out_page_num;  //修改swap table对应模块，给它相应位置存入被swap出来的那个page number
          vm->buffer[LRU_index * 32 + offset] = value;  // 修改physical memory中相应位置的值
        }
        else{                           //disk里找到了
          exchange_mem_disk_content(vm, LRU_index, disk_index);   // 交换memory和disk中的block的值
          vm->invert_page_table[LRU_index] &= 0x00000000;    // 修改page table
          vm->invert_page_table[LRU_index] |= (u32)vpn<<16; 
          (vm->invert_page_table[LRU_index])++;  
          vm->swap_table[disk_index] &= 0x00000000;    
          vm->swap_table[disk_index] |= out_page_num;  //修改swap table对应模块，给它相应位置存入被swap出来的那个page number
          vm->buffer[LRU_index * 32 + offset] = value;   // 修改内存中的对应地址值
        }
      }
    }
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  int value;
  for (int i = 0; i < input_size / 4; i++) {
		value = vm_read(vm, (i + offset) * 4 + threadIdx.x);
    results[i * 4 + threadIdx.x] = value;
	}
  
}

