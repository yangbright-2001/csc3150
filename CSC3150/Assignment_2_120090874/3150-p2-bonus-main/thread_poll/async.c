#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//上面的是自己加的
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

static my_queue_t *thread_pool = NULL;

void async_init(int num_threads) {
    thread_pool = calloc(1, sizeof(my_queue_t));
    // pthread_t thread[num_threads];
    int rc;
    int i;

    thread_pool->head = NULL;
    thread_pool->shutdown = 0;

    pthread_mutex_init(&(thread_pool->lock), NULL);
    pthread_cond_init(&(thread_pool->condition), NULL);

    thread_pool -> thread_id = malloc(num_threads * sizeof(pthread_t));

    for(i = 0; i < num_threads; i++){
		rc = pthread_create(&(thread_pool -> thread_id[i]), NULL, execute_routine, NULL); 
		if(rc){
			printf("ERROR in creating thread for logs_move: return error number is %d", rc);
		}
	}

    return;
    /** TODO: create num_threads threads and initialize the thread pool **/
}

void async_run(void (*hanlder)(int), int args) {
    // hanlder(args);
    /** TODO: rewrite it to support thread pool **/
    my_item_t *task, *element;
    task = malloc(sizeof(my_item_t));
    
    task->routine = hanlder;
    task->arg = (long)args;
    task->next = NULL;

    pthread_mutex_lock(&(thread_pool->lock)); 
    element = thread_pool->head; 
    if (!element){   //if no element in hte original task list
        thread_pool->head = task;   
    }
    else{
        while(element->next){
            element = element->next;
        }
        element->next = task;
    }
    pthread_cond_signal(&(thread_pool->condition));  //send signal to wake up the threads
    pthread_mutex_unlock(&(thread_pool->lock));
    
}

static void* execute_routine(void *arg){
    my_item_t *task;
    while(1){
        pthread_mutex_lock(&(thread_pool -> lock));
        while ((!(thread_pool->head)) && (!(thread_pool->shutdown))){   //if there is no tasks and the thread is not shut down
            pthread_cond_wait(&(thread_pool->condition), &(thread_pool->lock));
        }
        if (thread_pool -> shutdown){   
            pthread_mutex_unlock(&(thread_pool->lock));
            pthread_exit(NULL);
        }
        task = thread_pool->head;  //the thread get the task
        thread_pool->head = thread_pool->head->next;  //update the task queue
        pthread_mutex_unlock(&thread_pool->lock);
        task->routine(task->arg);  //execute the task
        free(task);

    }
}

