#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>

typedef struct my_item {  //任务
  /* TODO: More stuff here, maybe? */
  void* (*routine)(void*);  //task funciton
  void *arg;  //parameter of the task function
  struct my_item *next;
  // struct my_item *prev;
} my_item_t;

typedef struct my_queue { //进程池
  // int size;
  /* TODO: More stuff here, maybe? */
  int shutdown;
  pthread_t *thread_id; //线程id数组
  my_item_t *head;  //任务链表
  pthread_mutex_t lock;
  pthread_cond_t condition;
} my_queue_t;




void async_init(int);
void async_run(void (*fx)(int), int args);
static void* execute_routine(void *arg);

#endif
