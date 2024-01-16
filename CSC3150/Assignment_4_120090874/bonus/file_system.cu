#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  for(int i = 0; i < VOLUME_SIZE; i++){   //自己加的，初始化volumn数组，全部初始化为0
    volume[i] = 0;    //这个不是数字0，这个是空(它的ascii码是0)...
  }
  for(int i = 0; i < 1024; i++){
    fs->volume[4096 + 32 * i + 21] = 1;   //初始化时，empty bit全部设为“1”（1表示空，0表示非空）
  }                                         //应该是设成1，还是‘1’啊

}

//在FCB里面找此文件名是否存在，并返回其所在的那个FCB的index
__device__ int search_file_name(FileSystem *fs, char *s){
  int FCB_index = -1;
  for(int i = 0; i < 1024; i++){  //遍历这1024个block
    if ( (fs->volume[4096 + 32 * i + 21]) != 1 ){   //这一个FCB非空，我们才去搜索它
      for (int j = 0; j < 20; j++){             //因为文件最大长度是20
        if ( ((fs->volume[4096 + 32 * i + j]) != '\0') && (s[j] != '\0') )  {      //FCB的文件名与要搜索的文件名对应位置都非‘\0'
          if ( s[j] != (fs->volume[4096 + 32 * i + j]) ){       //没匹配上，退出当前循环对这个FCB的搜索
            break;
          }
        }
        else if ( ((fs->volume[4096 + 32 * i + j]) == '\0') && (s[j] == '\0')) {    //匹配上了（因为默认要搜索的文件名s不是空的） 
          FCB_index = i;    
          break;
        }
        else{   //FCB里面的名字或者要搜索的文件名遇到'\0'结束符提前结束了，那显然没匹配上，也退出对当前FCB的搜索
          break;
        }       
      }
    }
    if (FCB_index != -1){ //匹配上文件名了，那就可以退出最外层的循环了（不用再往下找FCB了）
      break;
    }
  }
  return FCB_index; 
}

//把bitmap批量从0改成1，或从1改成0
__device__ void modify_bitmap(FileSystem *fs, int start_index, int block_num, int mode){  
  int byte, offset;
  int start = start_index;
  //把bitmap从0改成1
  if (mode == 0){   
    for(int i = 0; i < block_num; i++){    
      byte = start / 8;
      offset = start % 8;   //start是当前文件的stock_index
      fs->volume[byte] |= 1 << offset;   //更新VCB这个bitmap
      start += 1;
    }
  }

  //把bitmap从1改成0
  if (mode == 1){
    for(int i = 0; i < block_num; i++){    
      int byte = start / 8;
      int offset = start % 8;   //start是当前文件的stock_index
      fs->volume[byte] &= ~(1 << offset);   
      start += 1;
    }
  }
  
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  int FCB_index, empty_FCB;
  u32 ptr = 0x00000000;
  int block_index = -1;
  if (op == G_READ){  //32位的二进制数最前面两位，read是10
    ptr |= 0x80000000;
    FCB_index = search_file_name(fs, s);
    ptr |= FCB_index;
  }
  else{  
    ptr |= 0x40000000; //32位的二进制数最前面两位，write是01
    FCB_index = search_file_name(fs, s);
    if (FCB_index != -1){   //找得到这个文件名
      ptr |= FCB_index;
    }

    else{   //找不到此文件，那就只好新建一个0-byte file,找一个空FCB格子并填入有关信息，更新VCB这个bitmap
      //找一个空的FCB格子
      for(int i = 0; i < 1024; i++){  //找一个空的FCB格子记录这个新建的文件信息
        if ( (fs->volume[4096 + 32 * i + 21]) == 1 ){
          empty_FCB = i;
          break;
        }
      }

      //找一个空的storage block，更新FCB
      for(int i = 0; i < 4096; i++){  //找一个空的storage block格子，从superblock这个bitmap里面找0
        for(int j = 0; j < 8; j++){   //这些都是index（从0开始的）
          if ( ( (fs->volume[i]) >> j & 1 ) == 0 ){     //找一个二进制数中从右往左数的第一个0: (number >> position) & 1      
            block_index = 8 * i + j;
            fs->volume[i] |= 1 << j;   //更新bitmap
            break;   //找到了，就立刻退出当前循环
          }
        }
        if (block_index != -1){ //找到了，就退出最外层循环
          break;  
        }
      }

      int length = 0;
      for (int i = 0; i < 20; i++){
        if (s[i] != '\0'){
          fs->volume[4096 + 32 * empty_FCB + i] = s[i];
          length += 1;
        }
        else{
          break;
        }
      }
      fs->volume[4096 + 32 * empty_FCB + length] = '\0';

      fs->volume[4096 + 32 * empty_FCB + 20] = 0;   //是文件，这个bit是0，是目录的话，这个bit是1
      
      fs->volume[4096 + 32 * empty_FCB + 21] = 0;   //更新这个FCB的empty bit为0（1是空，0是非空）

      fs->volume[4096 + 32 * empty_FCB + 22] = block_index / 256;  //更新start position （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 23] = block_index % 256; 

      fs->volume[4096 + 32 * empty_FCB + 24] = gtime / 256;  //更新create time （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 25] = gtime % 256; 

      fs->volume[4096 + 32 * empty_FCB + 26] = gtime / 256;  //更新modified time （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 27] = gtime % 256; 

      //file_size应该不用更新了，因为一开始就是0（但可能要在remove的时候把FCB清干净）,算了，保险起见，还是赋值成0吧
      //虽然这个文件占据了一个storage block格子，但它实际存的文件大小是0
      fs->volume[4096 + 32 * empty_FCB + 28] = 0;
      fs->volume[4096 + 32 * empty_FCB + 29] = 0;  //更新modified time （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 30] = 0; 
      fs->volume[4096 + 32 * empty_FCB + 31] = 0;

      gtime++;
      ptr |= empty_FCB;

    }
  }
  return ptr;   //返回的是记录有读写权限以及FCB的位置的一个32位二进制数
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  int file_size, FCB_index;
  int block_index = 0;   //它所存在的storage block的index
  int actual_address;   //它在volume中实际的位置（以byte为单位之索引）
  //检查写进来的fp这个32位二进制数是否合法，这其中包括是否具有read权限，以及需要read出来的大小是否超出了文件本身的大小
  if (fp >> 31 == 0){  //read模式，fp最高两位是10
    printf("The file pointer has no access to read!\n");
    return;
  }
  //检查需要read出来的byte的大小是否已经超出了文件大小范围
  FCB_index = (fp & 0x3FF);   //取出这个文件FCB存放的位置（fp最后10位）

  //int bit_1 = fs->volume[4096 + 32 * FCB_index + 28];   //file size这个二进制数的最高八位
  int bit_2 = fs->volume[4096 + 32 * FCB_index + 29];
  int bit_3 = fs->volume[4096 + 32 * FCB_index + 30];
  int bit_4 = fs->volume[4096 + 32 * FCB_index + 31];   //file size这个二进制数的最末八位
  // file_size = bit_1 * 1677216 + bit_2 * 65536 + bit_3 * 256 + bit_4;   //3byte存file size其实是够的，所以bit_1必是0
  file_size = bit_2 * 65536 + bit_3 * 256 + bit_4;   //3byte存file size其实是够的，所以bit_1必是0

  // printf("read from FCB index %d\n", FCB_index);
  // printf("FCB_index is %d, each bit is %d, %d and %d, size of the file is %d\n", FCB_index, bit_2, bit_3, bit_4, file_size);  //测试代码

  if (size > file_size){
    printf("The size you want to read has exceeded the original file size!\n");
    return;
  }


  if (file_size == 0){  //我们想读的这个file是空的（大小是0byte），说明这个file是fs_open刚创建的
    return;
  }

  block_index += fs->volume[4096 + 32 * FCB_index + 22];
  block_index *= 256;
  block_index += fs->volume[4096 + 32 * FCB_index + 23];    //算出该文件所在的storage block的block index（从0开始记）
  actual_address = fs->FILE_BASE_ADDRESS + block_index * fs->STORAGE_BLOCK_SIZE;  //根据storage block的index算出实际的address位置（以byte为单位）
  // uchar a;

  //这是原先的read代码
  for(int i = 0; i < size; i++){
    output[i] = 0;
    output[i] = (fs->volume[actual_address + i]);
    //printf("%c\n", fs->volume[actual_address + i]);
    //printf("%c\n",output[i]);
  }


}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  int original_size, FCB_index;
  int block_index = 0;
  int start_address;  //初始那个block对应的地址
  int block_need, remainder, block_byte;    //要存这么多内存所需要的storage block的数量(可能不是整数，所以需要一个remainder),以及他们对应多少个byte
  int current_start = 0;  //初始的start_address
  int empty_block_num;   //因为给空file分配的那个内存block也算一个
  int byte_index, offset;

  if ( (fp >> 30 & 1 ) == 0 ){      //检查是否有写的权限
    printf("The file pointer has no access to write!\n");
    return -1;
  }
  if (size > fs->MAX_FILE_SIZE){        //检查要写的这个文件大小是否已超过了1024KB这个最大限制
    printf("The size written has exceeded the largest file size limit!\n");
    return -1;
  }

  FCB_index = (fp & 0x3FF);   //从fp中取出这个文件FCB存放的位置信息（fp最后10位）
  //int bit_1 = fs->volume[4096 + 32 * FCB_index + 28];   //file size这个二进制数的最高八位
  //printf("write to FCB_index %d\n",FCB_index);
  //printf("write from FCB index %d\n", FCB_index);
  int bit_2 = fs->volume[4096 + 32 * FCB_index + 29];
  int bit_3 = fs->volume[4096 + 32 * FCB_index + 30];
  int bit_4 = fs->volume[4096 + 32 * FCB_index + 31];   //file size这个二进制数的最末八位
  // file_size = bit_1 * 1677216 + bit_2 * 65536 + bit_3 * 256 + bit_4;   //3byte存file size其实是够的，所以bit_1必是0
  original_size = bit_2 * 65536 + bit_3 * 256 + bit_4;   //3byte存file size其实是够的，所以bit_1必是0

  block_index += fs->volume[4096 + 32 * FCB_index + 22];
  block_index *= 256;
  block_index += fs->volume[4096 + 32 * FCB_index + 23];    //统一算出该文件原先start所所在的storage block的block index（从0开始记）

  start_address = fs->FILE_BASE_ADDRESS + block_index * fs->STORAGE_BLOCK_SIZE;  

  //如果初始大小不为0，则写入前需要把原先的内容删干净,在VCB里把除了start的那个block以外的bit值设为0
  if (original_size != 0){
    // int original_start_address = FILE_BASE_ADDRESS + block_index * STORAGE_BLOCK_SIZE;
    //把原先的block的内容清空
    for(int i = 0; i < original_size; i++){        
        fs->volume[start_address + i] = 0x00;
    }

    //修改bitmap，它的start那个block先保留1
    int block_num = original_size / 32;
    int remainder_1 = original_size % 32;
    if (remainder_1 > 0){
      block_num += 1;
    }
  
    modify_bitmap(fs, block_index + 1, block_num - 1, 1);

  }  

  //在这里统一更新gtime
  fs->volume[4096 + 32 * FCB_index + 26] = gtime / 256;  //更新modified time （2B，第一个byte存高八位，第二个byte存低八位）
  fs->volume[4096 + 32 * FCB_index + 27] = gtime % 256; 

  gtime++;    //更新gtime

  //在这里统一更新file_size参数
  fs->volume[4096 + 32 * FCB_index + 31] = size % 256;  //比如大小是65537个byte，那在这三个格子里的储存方式是101（1*65536+0*256+1）
  fs->volume[4096 + 32 * FCB_index + 30] = size / 256 % 256;
  fs->volume[4096 + 32 * FCB_index + 29] = size / 65536 % 256;

  if (size <= 32){    //因为我们也给空文件分配了一个block的，所以小于32byte的内容可以塞得进去
    
    for(int i = 0; i < size; i++){        //把内容写入volume的file content里
      fs->volume[start_address + i] = 0x00;
      fs->volume[start_address + i] = input[i];
      // printf("%c\n",fs->volume[start_address + i]);  //测试
    }
    
  }

  else{       //要写入这个空文件里面的内容大于32，可能需要compact了
    block_need = size / 32;
    remainder = size % 32;
    if (remainder > 0){
      block_need += 1;    //有余数，意味着要多加一个格来存
    }

    //然后找这个位置后面有几个0（几个空的storage block）
    empty_block_num = 1;  //因为给空file分配的那个内存block也算一个

    for (int i = block_index + 1; i < 32768; i++){
      byte_index = i / 8;  //在VCB的哪个byte
      offset = i % 8;     //具体从右往左数第几个位置
      if (fs->volume[byte_index] >> offset & 1 == 1){
        break;
      }
      empty_block_num += 1;
    }

    //block足够，不用compact，直接写
    if (empty_block_num >= block_need){     
      
      //把内容写入volume的file content里
      for(int i = 0; i < size; i++){ 
        fs->volume[start_address + i] = 0x00;       
        fs->volume[start_address + i] = input[i];
      }

      modify_bitmap(fs, block_index, block_need, 0);
    }

    //当前block_index位置不够啊，需要compact
    else{     
      int hole;    //VCB中第一个0的index
      int search_index = 0;
      int next_file_loc;
      int byte, off, byte_0, off_0;
      int need_compaction;
      int flag;
      
      //判断是否需要compact，并compact
      while(search_index < 32768){  
        //printf("search_index=%d\n",search_index);
        need_compaction = 0;   //是否还需要compact(等于-1意味着flag=0，然后退出整个while循环)
        // need_compaction = -1;   //是否还需要compact
        flag = 1;
        byte_index = search_index / 8;  //在VCB的哪个byte
        offset = search_index % 8;     //具体从右往左数第几个位置

        if (  (  (fs->volume[byte_index] >> offset & 1) == 0  ) || ( search_index == block_index )  ){   //找到第一个0，意味着有可能要开始compact（也可能不用，比如后面全是0）
          hole = search_index;   //找到第一个0所在的index
          need_compaction = -1;  //存在退出整个while循环的可能性
          for (int i = hole + 1; i < 32768; i++){    //接着往后找，找找看有没有非0（看看后面有没地方还存了文件的）
            byte_0 = i / 8;  //在bitmap的哪个byte
            off_0 = i % 8;     //具体从右往左数第几个位置
            // need_compaction = -1;

            if ( ( (fs->volume[byte_0] >> off_0 & 1) == 1 ) && ( i != block_index )  ){  //hole这个空格后面有存文件的地方，所以需要compact！
              next_file_loc = i;   //下一个存放有文件的block的index，要开始compact
              need_compaction = 1; //需要compact

              //找FCB中start在这个index位置的文件对应的FCB'
              int target_index, start;
              for (int j = 0; j < 1024; j++){
                if ( (fs->volume[4096 + 32 * j + 21]) != 1 ){   //这一个FCB非空，我们才去搜索它
                  start = 0;
                  start += (fs->volume[4096 + 32 * j + 22] * 256);
                  start += fs->volume[4096 + 32 * j + 23];

                  if (start == next_file_loc){  //找到了
                    target_index = j;  //存这个文件对应的FCB index
                    // printf("search index at %d, ",search_index);
                    // printf("find target_index: %d\n",target_index);  //测试
                    break;
                  }
                }
                
              }

              //更新一下这个文件FCB中对应的start
              fs->volume[4096 + 32 * target_index + 22] = hole / 256;
              fs->volume[4096 + 32 * target_index + 23] = hole % 256;

              //计算一下要挪过来的文件的size
              int target_size = 0;
              target_size += (fs->volume[4096 + 32 * target_index + 29] * 65536);
              target_size += (fs->volume[4096 + 32 * target_index + 30] * 256);
              target_size += (fs->volume[4096 + 32 * target_index + 31]);

              //把东西compact过来
              int from_address, to_address;
              for (int k = 0; k < target_size; k++){
                from_address = fs->FILE_BASE_ADDRESS + next_file_loc * (fs->STORAGE_BLOCK_SIZE);
                to_address = fs->FILE_BASE_ADDRESS + hole * (fs->STORAGE_BLOCK_SIZE);
                fs->volume[to_address + k] = 0x00;
                fs->volume[to_address + k] = fs->volume[from_address + k];  //把东西挪过来
                fs->volume[from_address + k] = 0;
              }

              //改一下bitmap
              int occupied_block = target_size / 32;
              if (target_size % 32 != 0){
                occupied_block += 1;  //这个文件占据的block的数量
              }
              
              // modify_bitmap(fs, hole, occupied_block, 0);  //把0的block变成1
              //注意先后顺序！先变成0，再变1。如果是有重合的时候，比如011变成110，顺序反了会出大问题（会由011变成100）
              modify_bitmap(fs, next_file_loc, occupied_block, 1);  //把1的block变成0 
              modify_bitmap(fs, hole, occupied_block, 0);  //把0的block变成1
              //我们最后处理需要写入的这个文件，把它放到大家的后面                
              break;  //一次compact（挪）一个文件，因为hole不一样，所以compact一次之后就要break

            }

          }
          if (need_compaction == -1){
            flag = 0;
          }
        }

        search_index += 1;
        // if (need_compaction == -1){
        //   break;
        // }
        if (flag == 0){
          break;
        }

      }

      //compact完毕后，我们把要写入的文件放到所有文件的最后
      //找第一个空的格子
      int new_index;
      for(int i = 0; i < 32768; i++){
        byte = i / 8;
        off = i % 8;
        if ( (fs->volume[byte] >> off & 1) == 0 ){  //找到新的“block_index”
          new_index = i;
          break;
        }          
      }
      //printf("new index is %d\n", new_index);
      
      //更新start
      fs->volume[4096 + 32 * FCB_index + 22] = new_index / 256;
      fs->volume[4096 + 32 * FCB_index + 23] = new_index % 256; 

      //算出新的地址并写入   
      int new_address = fs->FILE_BASE_ADDRESS + new_index * fs->STORAGE_BLOCK_SIZE;
      //printf("%d\n", new_address);
      for(int i = 0; i < size; i++){        //把内容写入volume的file content里
        fs->volume[new_address + i] = 0;
        fs->volume[new_address + i] = input[i];
      }

      modify_bitmap(fs, new_index, block_need, 0);

    }
  } 

  return 0;
  // }

}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  // if ( (op != LS_D) && (op != LS_S) ){
  //   return;
  // }

  int arr[1024];
  int index = 0;  //array的index
  //把FCB的index作为数组元素放进array中
  for (int i = 0; i < 1024; i++){
    arr[i] = 0;
  }
  for(int i = 0; i < 1024; i++){
    if (fs->volume[4096 + 32 * i + 21] != 1){   //如果非空，就放进array里
      arr[index] = i;
      index += 1;
    }
  }

  uchar name[20];
  //按修改时间来list（最近修改的先列出来）
  int size;
  if (op == LS_D){
    int modified_time_1, modified_time_2, FCB_index_1, FCB_index_2;
    int tmp;
    printf("===sorted %d files by modified time===\n", index);

    for(int i = 0; i < index - 1; i++){
      for (int j = 0; j < index - i - 1; j++){
        FCB_index_1 = arr[j];
        FCB_index_2 = arr[j + 1];
        modified_time_1 = 0;
        modified_time_1 += (fs->volume[4096 + 32 * FCB_index_1 + 26]) * 256;
        modified_time_1 += fs->volume[4096 + 32 * FCB_index_1 + 27];
        modified_time_2 = 0;
        modified_time_2 += (fs->volume[4096 + 32 * FCB_index_2 + 26]) * 256;
        modified_time_2 += fs->volume[4096 + 32 * FCB_index_2 + 27];
        //printf("time 1 is %d, time 2 is %d\n", modified_time_1, modified_time_2);
        if (modified_time_1 < modified_time_2){
          //printf("swap %d and %d\n", arr[j], arr[j+1]);
          //printf("time 1 is %d, time 2 is %d\n", modified_time_1, modified_time_2);
        //   swap = 1;
        // }
        // if (swap == 1){
          int tmp = FCB_index_1;
          arr[j] = FCB_index_2;
          arr[j + 1] = tmp;
        }
      }
    }
    
    //打印出来
    for (int i = 0; i < index; i++){  //据说不要一个一个字符打
      // uchar * ptr = &(fs->volume[4096 + 32 * arr[i]])
      for (int k = 0; k < 20; k++){
        name[k] = 0;
      }
      for (int j = 0; j < 20; j++){
        if (fs->volume[4096 + 32 * arr[i] + j] != '\0'){
          name[j] = (fs->volume[4096 + 32 * arr[i] + j]);
          // printf("%c", (char)fs->volume[4096 + 32 * arr[i] + j]);
        }
        else{
          break;
        }
      }
      if (fs->volume[4096 + 32 * arr[i] + 29] == 0){  
        printf("%s\n", name);
        printf("  d\n");
      }
      size = 0;
      size += (fs->volume[4096 + 32 * arr[i] + 29] * 65536);
      size += (fs->volume[4096 + 32 * arr[i] + 30] * 256);
      size += (fs->volume[4096 + 32 * arr[i] + 31])
  
      if (fs->volume[4096 + 32 * arr[i] + 20] == 1){    //目录
        printf("%s   %d\n", name, size);
        printf("  d\n");      //如果是目录，那就打个d在后面
      } 
      else{     //普通文件
        printf("%s   %d\n", name, size);
      }
    }

  }

  //按文件大小来list（文件大小相同时，按create time排）
  else if (op == LS_S){
    int size_1, size_2, create_time_1, create_time_2, FCB_index_1, FCB_index_2;
    int tmp;
    printf("===sort %d files by file size===\n", index);
    
    for(int i = 0; i < index - 1; i++){
      for (int j = 0; j < index - i - 1; j++){
        int swap = 0;
        FCB_index_1 = arr[j];
        FCB_index_2 = arr[j + 1];
        size_1 = 0;
        size_2 = 0;
        size_1 += (fs->volume[4096 + 32 * FCB_index_1 + 29] * 65536);
        size_1 += (fs->volume[4096 + 32 * FCB_index_1 + 30] * 256);
        size_1 += (fs->volume[4096 + 32 * FCB_index_1 + 31]);
        size_2 += (fs->volume[4096 + 32 * FCB_index_2 + 29] * 65536);
        size_2 += (fs->volume[4096 + 32 * FCB_index_2 + 30] * 256);
        size_2 += (fs->volume[4096 + 32 * FCB_index_2 + 31]);
        create_time_1 = 0;
        create_time_2 = 0;
        create_time_1 += (fs->volume[4096 + 32 * FCB_index_1 + 24] * 256);
        create_time_1 += (fs->volume[4096 + 32 * FCB_index_1 + 25]);
        create_time_2 += (fs->volume[4096 + 32 * FCB_index_2 + 24] * 256);
        create_time_2 += (fs->volume[4096 + 32 * FCB_index_2 + 25]);
        if (size_1 < size_2){
          swap = 1;
        }
        else if (size_1 == size_2){         //first create first print(create_time小的在数组前面)
          if (create_time_1 > create_time_2){
            swap = 1;
          }
        }
        if (swap == 1){
          tmp = FCB_index_1;
          arr[j] = FCB_index_2;
          arr[j + 1] = tmp;
        }
      }
    }
    // uchar name[20];
    // for (int i = 0; i < 20; i++){
    //   name[i] = 0;
    // }
    for (int i = 0; i < index; i++){  
      for (int k = 0; k < 20; k++){
        name[k] = 0;
      }
      //据说不要一个一个字符打
      // uchar * ptr = &(fs->volume[4096 + 32 * arr[i]])
      for (int j = 0; j < 20; j++){
        if (fs->volume[4096 + 32 * arr[i] + j] != '\0'){
          name[j] = (fs->volume[4096 + 32 * arr[i] + j]);
          // printf("%c", (char)fs->volume[4096 + 32 * arr[i] + j]);
        }
        else{
          break;
        }
      }
      //printf("%s",name);
      size = 0;
      //size += fs->volume[4096 + 32 * arr[i] + 29] 
      size += (fs->volume[4096 + 32 * arr[i] + 29] * 65536);
      size += (fs->volume[4096 + 32 * arr[i] + 30] * 256);
      size += (fs->volume[4096 + 32 * arr[i] + 31]);
      //printf(" %d", size);
      if (fs->volume[4096 + 32 * arr[i] + 29] == 1){  //如果是目录，那就打个d在后面
        printf("%s  %d", name, size);
        printf("  d\n");
      } 
      else{
        printf("%s  %d", name, size);
      }

    }


  }

}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  if (op == RM){
    
    int FCB_index;
    int block_index = 0;
    FCB_index =search_file_name(fs, s);
    if (FCB_index == -1){
      printf("The file you want to remove does not exist!\n");
      return;
    }

    else if (fs->volume[4096 + 32 * FCB_index + 20] != 0 ){   //不能RM删除目录
      printf("You cannot use RM to delete a directory!\n");
      return;
    }

    else{
      //int bit_1 = fs->volume[4096 + 32 * FCB_index + 28];   //file size这个二进制数的最高八位
      int bit_2 = fs->volume[4096 + 32 * FCB_index + 29];
      int bit_3 = fs->volume[4096 + 32 * FCB_index + 30];
      int bit_4 = fs->volume[4096 + 32 * FCB_index + 31];   //file size这个二进制数的最末八位
      // file_size = bit_1 * 1677216 + bit_2 * 65536 + bit_3 * 256 + bit_4;   //3byte存file size其实是够的，所以bit_1必是0
      int file_size = bit_2 * 65536 + bit_3 * 256 + bit_4;   //3byte存file size其实是够的，所以bit_1必是0

      block_index += fs->volume[4096 + 32 * FCB_index + 22] * 256;
      block_index += fs->volume[4096 + 32 * FCB_index + 23];    //算出该文件原先start所所在的storage block的block index（从0开始记）

      int start_address = fs->FILE_BASE_ADDRESS + block_index * fs->STORAGE_BLOCK_SIZE;

      //把原先的block的内容清空
      for(int i = 0; i < file_size; i++){        
          fs->volume[start_address + i] = 0;
      }

      //修改bitmap
      int block_num = file_size / 32;
      int remainder = file_size % 32;
      if (remainder > 0){
        block_num += 1;
      }
  
      modify_bitmap(fs, block_index, block_num, 1);

      //把这个文件的FCB清0
      for (int i = 0; i < 32; i++){
        fs->volume[4096 + 32 * FCB_index + i] = block_index / 256;
      }
      fs->volume[4096 + 32 * FCB_index + 21] = 1;   //更新这个FCB的empty bit至1（1是空，0是非空）

    }  
  }

  else if (op == MKDIR){    //MKDIR
    //找一个空的FCB格子
      for(int i = 0; i < 1024; i++){  //找一个空的FCB格子记录这个新建的文件信息
        if ( (fs->volume[4096 + 32 * i + 21]) == 1 ){
          empty_FCB = i;
          break;
        }
      }

      //找一个空的storage block，更新FCB
      for(int i = 0; i < 4096; i++){  //找一个空的storage block格子，从superblock这个bitmap里面找0
        for(int j = 0; j < 8; j++){   //这些都是index（从0开始的）
          if ( ( (fs->volume[i]) >> j & 1 ) == 0 ){     //找一个二进制数中从右往左数的第一个0: (number >> position) & 1      
            block_index = 8 * i + j;
            fs->volume[i] |= 1 << j;   //更新bitmap
            break;   //找到了，就立刻退出当前循环
          }
        }
        if (block_index != -1){ //找到了，就退出最外层循环
          break;  
        }
      }

      int length = 0;
      for (int i = 0; i < 20; i++){
        if (s[i] != '\0'){
          fs->volume[4096 + 32 * empty_FCB + i] = s[i];
          length += 1;
        }
        else{
          break;
        }
      }
      fs->volume[4096 + 32 * empty_FCB + length] = '\0';

      fs->volume[4096 + 32 * empty_FCB + 20] = 1;   //是目录，这个bit是1
      
      fs->volume[4096 + 32 * empty_FCB + 21] = 0;   //更新这个FCB的empty bit为0（1是空，0是非空）

      fs->volume[4096 + 32 * empty_FCB + 22] = block_index / 256;  //更新start position （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 23] = block_index % 256; 

      fs->volume[4096 + 32 * empty_FCB + 24] = gtime / 256;  //更新create time （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 25] = gtime % 256; 

      fs->volume[4096 + 32 * empty_FCB + 26] = gtime / 256;  //更新modified time （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 27] = gtime % 256; 

      //file_size应该不用更新了，因为一开始就是0（但可能要在remove的时候把FCB清干净）,算了，保险起见，还是赋值成0吧
      //虽然这个文件占据了一个storage block格子，但它实际存的文件大小是0
      fs->volume[4096 + 32 * empty_FCB + 28] = 0;
      fs->volume[4096 + 32 * empty_FCB + 29] = 0;  //更新modified time （2B，第一个byte存高八位，第二个byte存低八位）
      fs->volume[4096 + 32 * empty_FCB + 30] = 0; 
      fs->volume[4096 + 32 * empty_FCB + 31] = 0;

      gtime++;
      
  }
  else if (op == CD){
    int FCB_index = search_file_name(fs, s);
    if (FCB_index == -1){
      printf("No such directory!\n");
      return;
    }
    else{
      
    }
  }

}
