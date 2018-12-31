#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MT.h"

/*定数定義*/

/*モデル*/
#define TATE 12
#define YOKO 12
#define STATE_NO TATE * YOKO
#define ACTION_NO 4
#define START_S_X 1
#define START_S_Y 1
#define GORL_S_X 8
#define GORL_S_Y 8
#define ACTION_UP 0
#define ACTION_DOWN 1
#define ACTION_LEFT 2
#define ACTION_RIGHT 3
#define X 0
#define Y 1
#define SEED 10

/*強化学習*/
#define ALPHA 0.9
#define EPSILON 0.3
#define GAMMA 0.9
#define EPISODE_NO 10000
#define STEP_NO 1000

/*ニューラルネット*/
#define LIMIT 0.3
#define NNLIMIT 0.01
#define NNALPHA 0.01
#define DIGITS 4
#define INPUT_UNIT_NO TATE + YOKO 
#define MID_UNIT_NO 8
#define OUTPUT_UNIT_NO ACTION_NO

extern void initW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
extern void printW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
extern void printQ(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]);
void getQ(int s[], double Q[OUTPUT_UNIT_NO], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]);
void initinput(int input[INPUT_UNIT_NO]);
void setinput(int input[], int s[]);
void calcmidunit(double result[], int input_to_unit[], double w[MID_UNIT_NO][INPUT_UNIT_NO + 1]);
void calcoutunit(double result[], double input_to_unit[], double w[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
double sigmoidfunc(double z);
double sigmoiddash(double y);
double tanhfunc(double z);
double tanhdash(double y);
extern int pi(int s[], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]);
int argmaxQ_a(double Q[OUTPUT_UNIT_NO]);
extern void statetransition(int s[], int a, int s_next[]);
extern double reword(int s[], int s_next[]);
extern void learning_units(int s[], int a, double r, int s_next[], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]);
void make_teacher_data(int a, double t[OUTPUT_UNIT_NO], double Q[OUTPUT_UNIT_NO], double maxQnext, double r);
double updateQvalue(double Q, double Qnext, double r);
void linerconv(double Q[], double t[]);
void bp_for_outunit(double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], double unit_err[OUTPUT_UNIT_NO], double Q[OUTPUT_UNIT_NO], double t[OUTPUT_UNIT_NO]);
void bp_for_midunit(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[OUTPUT_UNIT_NO], double result_mid[MID_UNIT_NO], double out_unit_err[OUTPUT_UNIT_NO]);
double errsum(double Q[OUTPUT_UNIT_NO], double t[OUTPUT_UNIT_NO]);
double errcalc(int s[], int a, double r, int s_next[], int a_next, double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]);
void write_reward_to_file(double reward, int t, int episode_count);
