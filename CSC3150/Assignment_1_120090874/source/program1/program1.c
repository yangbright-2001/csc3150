#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[])
{
	pid_t pid;
	int status;

	/* fork a child process */
	printf("Process start to fork\n");
	pid = fork();

	if (pid == -1) {
		perror("fork");
		exit(1);
	}

	/* execute test program */
	else {
		//child process
		if (pid == 0) {
			int i;
			char *arg[argc];

			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			printf("I'm the child process, my pid = %d\n",
			       getpid());
			printf("Child process start to execute test program:\n");
			// raise(SIGCHLD);
			execve(arg[0], arg, NULL);

			printf("Continue to run original child process, fail to execute the test program!\n");
			perror("execve");
			exit(EXIT_FAILURE);
		}
		//parent process
		/* wait for child process terminates */
		else {
			printf("I'm the parent process, my pid = %d\n",
			       getpid());
			waitpid(pid, &status, WUNTRACED);
			/* check child process'  termination status */
			printf("Parent process receives SIGCHLD signal\n");
			// if(WIFEXITED(status)){		//进程正常结束,输出进程退出参数（exit时的参数，比如exit(5)，则为5）
			// 	printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
			// }
			// else if(WIFSIGNALED(status)){	//进程异常终止,输出使得进程终止的信号编号,该编号即为signal对应之数字，比如信号SIGABRT对应数字6
			// 	printf("CHILD EXECUTION FAILED: %d\n",WTERMSIG(status));
			// }
			// else if(WIFSTOPPED(status)){	//进程处于暂停状态，输出使得进程暂停的信号编号
			// 	printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
			// }
			// else{		//WIFCONTINUED(status), 非0表示暂停后已经继续运行
			// 	printf("CHILD PROCESS CONTINUED\n");
			// }
			if (WIFSIGNALED(status)) {
				switch (WTERMSIG(status)) {
				case 6: {
					printf("child process get SIGABRT signal\n");
					break;
				}
				case 14: {
					printf("child process get SIGALRM signal\n");
					break;
				}
				case 7: {
					printf("child process get SIGBUS signal\n");
					break;
				}
				case 8: {
					printf("child process get SIGFPE signal\n");
					break;
				}
				case 1: {
					printf("child process get SIGHUP signal\n");
					break;
				}
				case 4: {
					printf("child process get SIGILL signal\n");
					break;
				}
				case 2: {
					printf("child process get SIGINT signal\n");
					break;
				}
				case 9: {
					printf("child process get SIGKILL signal\n");
					break;
				}
				case 13: {
					printf("child process get SIGPIPE signal\n");
					break;
				}
				case 3: {
					printf("child process get SIGQUIT signal\n");
					break;
				}
				case 11: {
					printf("child process get SIGSEGV signal\n");
					break;
				}
				case 15: {
					printf("child process get SIGTERM signal\n");
					break;
				}
				case 5: {
					printf("child process get SIGTRAP signal\n");
					break;
				}
				}
			} else if (WIFEXITED(status)) {
				printf("Normal termination with EXIT STATUS = %d\n",
				       WEXITSTATUS(status));
			} else if (WIFSTOPPED(status)) {
				printf("child process get SIGSTOP signal\n");
			}
			exit(0);
		}
	}
}
