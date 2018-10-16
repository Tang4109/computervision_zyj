#include <iostream>

using namespace std;
int flag = 0;//定义标志变量，用于实现循环输入
int route[100];//定义路径数组，用于存储路径
int time[100];//电梯所走的时间，存放入一个数组中
int n_num;//定义计算的次数
int sum = 0;//定义一个累加变量
int main()
{
	int N = 0;
	while (1) {
		cin >> N; //输入该行将输入的数字个数
			n_num += 1;//记录输入N的次数，即将要执行的计算次数
		if (N != 0) {

			while (flag < N) {
				flag += 1;
				cin >> route[flag - 1];//将路径存储进路径数组中

			}
			flag = 0;//标志变量归0,以便作为下一次输入路径的标志变量
			//计算结果
			int j = 0;
			sum = 6 * route[j] + 5;//先计算初始0层到第一个位置的时间
			for (j = 0; j < N - 1; j++) {//小于N-1是因为else if中的j+1刚好达到边界

				if (route[j] < route[j + 1]) {//计算上升时间
					sum = sum + 6 * (route[j + 1] - route[j]) + 5;
				}
				else if (route[j] > route[j + 1]) {//计算下降时间
					sum = sum + 4 * (route[j] - route[j + 1]) + 5;
				}

			}

			time[n_num - 1] = sum;//将累加变量sum的值放入时间数组中存储起来，最后一起输出
			sum = 0;//累加变量归0，以便进行下次计算

		}

		else {//输入N=0则结束输入，输出计算结果
			break;//跳出循环

		}
	}
	for (int i = 0; i < n_num - 1; i++) {
		cout << time[i] << endl;//输出时间

	}

	cin.get();
	cin.get();
	return 0;
}