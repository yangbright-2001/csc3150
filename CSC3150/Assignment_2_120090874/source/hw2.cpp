#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 

pthread_mutex_t mutex;

struct Node{
	int x , y; 
	Node(int _x, int _y) : x( _x ), y( _y ) {}; 
	Node(){}; 
} frog; 

int frog_on_log = 0;
int stop_process = 0;	//共有变量，控制frog_move, logs_move 进程
int flag = 0;
char map[ROW+10][COLUMN]; 
// int debug = 0;

//function prototype declaration
int kbhit(void);
void *frog_move(void *arg);
void *logs_move(void *t);
void *screen_render(void *arg);
// void judge_status(int flag);

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void *frog_move(void *arg){
	// int should_quit = 0;	//whether should quit the game(because of win/lose or user exit)
	char dir;
	while(!stop_process){
		if(kbhit()){
			dir = getchar();
			pthread_mutex_lock(&mutex);

			if (dir == 'q' || dir == 'Q'){	//quit
				// judge_status(1);
				flag = 1;
				// should_quit = 1;
				stop_process = 1;
			}

			if (dir == 'w' || dir == 'W'){	//up movement
				if ((frog.x - 1) == 0){	//have reach the top (the opposite bank) and win
					map[frog.x - 1][frog.y] = '0';
					map[frog.x][frog.y] = '=';
					frog.x = frog.x - 1;
					// judge_status(2);
					flag = 2;
					// should_quit = 1;
					stop_process = 1;
				}
				else if(frog.x == 10){
					if(map[frog.x - 1][frog.y] == '='){
						map[frog.x - 1][frog.y] = '0';
						map[frog.x][frog.y] = '|';
						frog.x = frog.x - 1;
					}
					else{
						// judge_status(3);
						flag = 3;
						// should_quit = 1;
						stop_process = 1;
					}						
				}
				else if(frog.x != 10 && frog.x != 1){
					if(map[frog.x - 1][frog.y] == '='){
						map[frog.x - 1][frog.y] = '0';
						map[frog.x][frog.y] = '=';
						frog.x = frog.x - 1;
					}
					else{
						// judge_status(3);
						flag = 3;
						// should_quit = 1;
						stop_process = 1;
					}
				}
			}

			if (dir == 's' || dir == 'S'){	//down movement
				if (frog.x != 10){
					if (map[frog.x + 1][frog.y] == '|' || map[frog.x + 1][frog.y] == '='){	
						map[frog.x + 1][frog.y] = '0';
						map[frog.x][frog.y] = '=';
						frog.x = frog.x + 1;
					}
					else{
						// judge_status(3);
						flag = 3;
						// should_quit = 1;
						stop_process = 1;
					}						
				}
			}

			if (dir == 'a' || dir == 'A'){	//left movement
				if (frog.x == 10){	
					if (frog.y != 0){
						map[frog.x][frog.y - 1] = '0';
						map[frog.x][frog.y] = '|';
						frog.y = frog.y - 1;
					}

				}else if (frog.x != 10){					
		
					frog.y = frog.y - 1;							
				}
			}	

			if (dir == 'd' || dir == 'D'){	//right movement
				if (frog.x == 10){	
					if (frog.y != 48){
						map[frog.x][frog.y + 1] = '0';
						map[frog.x][frog.y] = '|';
						frog.y = frog.y + 1;
					}

				}else if (frog.x != 10){					
					
					frog.y = frog.y + 1;											
				}
			}

			pthread_mutex_unlock(&mutex);
		}
	}

	pthread_exit(NULL);
} 


void *logs_move(void *t){	//参数t是log所在的行数（从0开始）

	int log_length = 15;
	long row_index;
	int i;
	int start;
	int current;
	row_index = (long)t;

	srand(time(0) + row_index * row_index * row_index * row_index);  //avoid having the too similar random seeds (start positions) for different logs
	start = rand() % 44;  //random start position index of the log: 0-43
	//start指的是火车头，往左走和往右走的火车头不一样（一个在左端一个在右端）

	while(!stop_process){
		pthread_mutex_lock(&mutex);
		for (i=0;i < 49; i++){
			map[row_index][i] = ' ';
		}
		if (row_index % 2 == 1){  //行数为从上往下数1,3,5，往左走, 火车头在最左
			if (row_index != frog.x){
				current = start;
				for (i = 0; i < log_length; i++){
					map[row_index][current] = '=';
					if (current == 48){
						current = 0;
					}
					else{
						current = current + 1;
					} 
				}
			}
			else{
				frog_on_log = 0;
				current = start;
				//frog_on_log = 0;
				for (i = 0; i < log_length; i++){
					if (frog.y < 0){
						//debug = 2;
						stop_process = 1;
						flag = 3;
						break;
					}
					if (current == frog.y){
						map[row_index][current] = '0';
						frog_on_log = 1;
					}
					else{
						map[row_index][current] = '=';
					}					
					if (current == 48){ //下一个位置的列坐标
						current = 0;
					}
					else{
						current = current + 1;
					} 					
				}
				if (frog_on_log){
					frog.y = frog.y - 1;
				}
				else{
					flag = 3;
					stop_process = 1;
				}
				//frog.y = frog.y - 1;
			}

			if (start == 0){
				start = 48;
			}
			else{
				start = start - 1;
			}
		}
		else{	//行数为从上往下数2,4,6，往右走,火车头在最右
			if (row_index != frog.x){
				current = start;
				for (i = 0; i < log_length; i++){
					map[row_index][current] = '=';
					if (current == 0){
						current = 48;
					}
					else{
						current = current - 1;
					} 
				}
			}
			else{
				frog_on_log = 0;  //initialize within every loop
				current = start;
				for (i = 0; i < log_length; i++){
					if (frog.y > 48){
						stop_process = 1;
						flag = 3;
						//debug = 2;
						break;
					}
					if (current == frog.y){
						map[row_index][current] = '0';
						frog_on_log = 1;
					}
					else{
						map[row_index][current] = '=';
					}					
					if (current == 0){
						current = 48;
					}
					else{
						current = current - 1;
					} 
				}
				if (frog_on_log){
					frog.y = frog.y + 1;
				}
				else{
					flag = 3;
					stop_process = 1;
				}
				//frog.y = frog.y + 1;
			}

			if (start == 48){
				start = 0;
			}
			else{
				start = start + 1;
			}
		}

		pthread_mutex_unlock(&mutex);
		usleep(100000);

	}

	pthread_exit(NULL);
}


void *screen_render(void *arg){
	
	while (!stop_process){
		pthread_mutex_lock(&mutex);
		printf("\033[H\033[2J");
		for(int i = 0;i <= 10;i ++ ){
			puts(map[i]);
		}
		pthread_mutex_unlock(&mutex);
		usleep(100000);
	}

	pthread_exit(NULL);
}

int main(int argc, char *argv[]){	
	int rc;
	long row_num;

	pthread_t frog_thread;
	pthread_t log_thread[9]; //threads for the movements of the 9 logs
	pthread_t screen_thread;

	//initialize the mutex
	pthread_mutex_init(&mutex, NULL);

	// Initialize the river map and frog's starting position
	memset(map, 0, sizeof(map));
	int i, j ; 
	for(i = 1; i < ROW; ++i){	
		for(j = 0; j < COLUMN - 1; ++j)	
			map[i][j] = ' ';  
	}	

	for(j = 0; j < COLUMN - 1; ++j)	//一共49个｜
		map[ROW][j] = map[0][j] = '|';

	for(j = 0; j < COLUMN - 1; ++j)	
		map[0][j] = map[0][j] = '|';

	frog = Node(ROW, (COLUMN-1)/2) ; 
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	for(i = 0; i <= ROW; ++i)	
		puts(map[i]);
	
	/*  Create pthreads for wood move and frog control.  */
	rc = pthread_create(&frog_thread, NULL, frog_move, NULL);
	if(rc){
		printf("ERROR in creating thread for frog_move: return error number is %d", rc);
	}

	for(row_num = 1; row_num < 10; row_num++){
		rc = pthread_create(&log_thread[row_num-1], NULL, logs_move, (void*)row_num);
		if(rc){
			printf("ERROR in creating thread for logs_move: return error number is %d", rc);
		}
	}

	rc = pthread_create(&screen_thread, NULL, screen_render, NULL);
	if(rc){
		printf("ERROR in creating thread for screen_render: return error number is %d", rc);
	}

	//join the threads
	pthread_join(frog_thread, NULL);
	for(row_num = 1; row_num < 10; row_num++){
		pthread_join(log_thread[row_num-1], NULL);
	}
	pthread_join(screen_thread, NULL);
	
	/*  Display the output for user: win, lose or quit.  */
	printf("\033[H\033[2J");
	if (flag == 1){		//exit
		printf("You exit the game.\n");
	}
	if (flag == 2){		//win
		printf("You win the game!!\n");
	}
	if (flag == 3){		//lose
		printf("You lose the game!!\n");
	}

	//destroy the mutex
    pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);

	return 0;

}