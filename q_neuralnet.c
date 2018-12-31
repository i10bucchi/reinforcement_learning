#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MT.h"

/*定数定義*/

/*モデル*/
#define TATE 13
#define YOKO 21
#define STATE_NO TATE * YOKO
#define ACTION_NO 4
#define START_S_X 1
#define START_S_Y TATE - 2
#define GORL_S_X YOKO - 2
#define GORL_S_Y TATE - 2
#define ACTION_UP 0
#define ACTION_DOWN 1
#define ACTION_LEFT 2
#define ACTION_RIGHT 3
#define X 0
#define Y 1
#define SEED 3

/*強化学習*/
#define ALPHA 0.9
#define EPSILON 0.3
#define GAMMA 0.9
#define EPISODE_NO 50000
#define STEP_NO 5000

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


int main(void) {
    int j, t;
    int s[2], a;
    int s_next[2], a_next;
    double r;
    double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1];
    double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1];
    double result_mid[MID_UNIT_NO];
    int input[INPUT_UNIT_NO];
    double err = 100.0;
    double rewardsum;

    init_genrand(SEED);

    // ニューラルネットの重み初期化
    initW(w_mid, w_out);
    // 初期設定の表示
    // printW(w_mid, w_out);
    printQ(w_mid, w_out, input, result_mid);

    //学習開始
    for (j = 0; j < EPISODE_NO; j++) {
        // 誤差の初期化
        err = 0.0;

        // 状態の初期化
        s[X] = START_S_X;
        s[Y] = START_S_Y;

        // 収益の初期化
        rewardsum = 0.0;

        for (t = 0; t < STEP_NO; t++) {
            // 政策piから行動を決定
            a = pi(s, w_mid, w_out, input, result_mid);;
            // a = pi(s, w_mid, w_out, input, result_mid);

            // 決定した行動をもとに次の状態へ遷移
            statetransition(s, a, s_next);

            // 遷移した事による報酬を観測
            r = reword(s, s_next);
            rewardsum += r;

            // Q値の更新(r + gammaQnextを教師データとした重みの更新)
            learning_units(s, a, r, s_next, w_mid, w_out, input, result_mid);

            // 状態を観測
            s[X] = s_next[X];
            s[Y] = s_next[Y];
            
            // ゴールしたら終了
            if (s[X] == GORL_S_X && s[Y] == GORL_S_Y) {
                break;
            }
        }
    }

    // printW(w_mid, w_out);
    printQ(w_mid, w_out, input, result_mid);
    printf("episode = %d\n", j);

    return 0;
}

void initW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i, j;
    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            w_mid[i][j] = genrand_real1() * 2 - 1;
        }
    }
    
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO + 1; j++) {
            w_out[i][j] = genrand_real1() * 2 - 1;
        }
    }
}

void printW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i, j;

    printf("###################### printw #########################\n");
    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            printf("%d -> %d: %lf\n", j, i, w_mid[i][j]);
        }
    }
    printf("-----------------------------------------------\n");
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO + 1; j++) {
            printf("%d -> %d: %lf\n", j, i, w_out[i][j]);
        }
    }
}

void printQ(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]) {
    int x, y, j;
    int s[2];
    int max_a;
    double Q[ACTION_NO];

    printf("###################### printQ #########################\n");

    // マップ表示
    for (y = 0; y < TATE; y++) {
        for (x = 0; x < YOKO; x++) {
            max_a = 0;
            s[X] = x;
            s[Y] = y;

            if (x == 0 || x == YOKO-1) {
                printf("|");
            }
            else if (y == 0 || y == TATE-1 || (y == TATE-2 && x > 1 && x < YOKO-2)) {
                printf("- ");
            }
            else {
                getQ(s, Q, w_mid, w_out, input, result_mid);
                
                for (j = 0; j < ACTION_NO; j++) {
                    if (Q[j] > Q[max_a]) {
                        max_a = j;
                    }
                }
                
                if (max_a == ACTION_UP) {
                    printf("↑ ");
                }
                else if (max_a == ACTION_DOWN) {
                    printf("↓ ");
                }
                else if (max_a == ACTION_LEFT) {
                    printf("← ");
                }
                else if (max_a == ACTION_RIGHT) {
                    printf("→ ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");
    
    // Q値表示
    // for (y = 1; y < TATE - 1; y++) {
    //     for (x = 1; x < YOKO - 1; x++) {
    //         printf("x, y = %d ,%d ", x, y);
    //         s[X] = x;
    //         s[Y] = y;
    //         getQ(s, Q, w_mid, w_out, input, result_mid);
    //         for (j = 0; j < ACTION_NO; j++) {
    //             printf("%lf ", Q[j]);
    //         }
    //         printf("\n");
    //     }
    // }
}

void getQ(int s[], double Q[OUTPUT_UNIT_NO], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]) {
    
    // 入力ニューロン設定
    initinput(input);
    setinput(input, s);
    
    // 中間層の計算
    calcmidunit(result_mid, input, w_mid);

    // 出力層の計算
    calcoutunit(Q, result_mid, w_out);

    // 入力ニューロン初期化
    initinput(input);
}

void initinput(int input[OUTPUT_UNIT_NO]) {
    int i;
    for (i = 0; i < INPUT_UNIT_NO + 1; i++) {
        input[i] = 0;
    }
}

void setinput(int input[], int s[]) {
    // 座標データを1, 0にして
    input[s[X]] = 1;
    input[TATE + s[Y]] = 1;

    // 生の座標データ
    
    //input[X] = s[X];
    //input[Y] = s[Y];

   // 生の座標データを2進数表記
   /*
    int i;

    for(i = DIGITS-1; i >= 0; i--) {
        input[i] = ((s[X] >> i) & 1);
    }

    for(i = (DIGITS * 2) - 1; i >= DIGITS; i--) {
        input[i] = ((s[Y] >> (i - DIGITS)) & 1);
    }
    */
    
}

void calcmidunit(double result[], int input_to_unit[], double w[MID_UNIT_NO][INPUT_UNIT_NO + 1]) {
    int i, j;
    double z;
    
    for (i = 0; i < MID_UNIT_NO; i++) {
        z = 0;
        for (j = 0; j < INPUT_UNIT_NO; j++) {
            z += input_to_unit[j] * w[i][j];
        }
        z += (-1) * w[i][j];
        result[i] = tanhfunc(z);
    }
}

void calcoutunit(double result[], double input_to_unit[], double w[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i, j;
    double z;
    
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        z = 0;
        for (j = 0; j < MID_UNIT_NO; j++) {
            z += input_to_unit[j] * w[i][j];
        }
        z -= w[i][j];
        result[i] = tanhfunc(z);
    }
}

double sigmoidfunc(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoiddash(double y) {
    return (y * (1 - y));
}

double tanhfunc(double z) {
    return tanh(z);
}

double tanhdash(double y) {
    return (4 / ((exp(y) + exp(-y)) * (exp(y) + exp(-y))));
}

int pi(int s[], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]) {
    double Q[ACTION_NO];

    getQ(s, Q, w_mid, w_out, input, result_mid);

    // イプシロンの確率でランダムに行動
    if (genrand_real1() < EPSILON) {
        return (genrand_int32() % ACTION_NO);
    }
    else {
        // 最大のQ値の行動を返す
        return argmaxQ_a(Q);
    }
}

int argmaxQ_a(double Q[OUTPUT_UNIT_NO]) {
    int i;
    int max_a;
    
    max_a = 0;
    for (i = 0; i < ACTION_NO; i++) {
        if (Q[i] > Q[max_a]) {
            max_a = i;
        }
    }
    return (max_a);
}

void statetransition(int s[], int a, int s_next[]) {
    if (a == ACTION_UP) {
        s_next[X] = s[X];
        s_next[Y] = s[Y] - 1;
    }
    else if (a == ACTION_DOWN) {
        s_next[X] = s[X];
        s_next[Y] = s[Y] + 1;
    }
    else if (a == ACTION_LEFT) {
        s_next[X] = s[X] - 1;
        s_next[Y] = s[Y];
    }
    else if (a == ACTION_RIGHT) {
        s_next[X] = s[X] + 1;
        s_next[Y] = s[Y];
    }

    // 壁に当たった場合
    if (s_next[X] == 0 || s_next[X] == YOKO - 1 || s_next[Y] == 0 || s_next[Y] == TATE - 1) {
        s_next[X] = s[X];
        s_next[Y] = s[Y];
    }
    // 崖だった場合
    if (s_next[Y] == TATE - 2 && s_next[X] != 1 && s_next[X] != YOKO - 2) {
        s_next[X] = s[X];
        s_next[Y] = s[Y];
    }
}

double reword(int s[], int s_next[]) {
    double r;
    // 壁の場合
    if (s_next[X] == s[X] && s_next[Y] == s[Y]) {
        r = -1.0;
    }
    // ゴールの場合
    else if (s_next[X] == GORL_S_X && s_next[Y] == GORL_S_Y) {
        r = 1.0;
    }
    else {
        r = -0.01;
    }

    return r;
}

void learning_units(int s[], int a, double r, int s_next[], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]) {
    int i;
    double maxQnext;
    double Q[OUTPUT_UNIT_NO], Qnext[OUTPUT_UNIT_NO];
    double t[OUTPUT_UNIT_NO];
    double co[OUTPUT_UNIT_NO], cm[MID_UNIT_NO]; // cost for output units error, cost for middle units error
    double next_result_mid[MID_UNIT_NO];
    double err = 100.00;

    while ((err) > NNLIMIT) {
        getQ(s, Q, w_mid, w_out, input, result_mid);
        getQ(s_next, Qnext, w_mid, w_out, input, next_result_mid);
        
        // 次状態における最大のQ値
        maxQnext = Qnext[argmaxQ_a(Qnext)];

        // 教師データ生成
        make_teacher_data(a, t, Q, maxQnext, r);

        // 出力層の学習(OUT -> MID)
        bp_for_outunit(w_out, result_mid, co, Q, t);

        initinput(input);
        setinput(input, s);

        // 中間層の学習(MID -> INPUT)
        bp_for_midunit(w_mid, w_out, input, result_mid, co);

        initinput(input);

        // 誤差を求める
        err = errsum(Q, t);

        // printf("unit learning err = %lf\n", err);
    }
}

void make_teacher_data(int a, double t[OUTPUT_UNIT_NO], double Q[OUTPUT_UNIT_NO], double maxQnext, double r) {
    int i;

    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        if (i == a) {
            t[i] = tanhfunc(updateQvalue(Q[a], maxQnext, r));
        }
        else {
            t[i] = Q[i];
        }
    }
}

double updateQvalue(double Q, double Qnext, double r) {
    // double updateQ;
    // updateQ = Q + (ALPHA * (r - Q + (GAMMA * Qnext)));
    // return (updateQ);

    double t = r + (GAMMA * Qnext);
    return (t);
}

void linerconv(double Q[], double t[]) {
    int i;
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        Q[i] *= 1;
        t[i] *= 1;
    }
}

void bp_for_outunit(double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], double unit_err[OUTPUT_UNIT_NO], double Q[OUTPUT_UNIT_NO], double t[OUTPUT_UNIT_NO]) {
    int i, j;

    // 出力層ユニット誤差取得
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        unit_err[i] = (Q[i] - t[i]) * tanhdash(Q[i]);
    }

    // 出力層の学習(OUT -> MID)
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO; j++) {
            w_out[i][j] -= NNALPHA * result_mid[j] * unit_err[i];
        }
        w_out[i][j] -= NNALPHA * (-1.0) * unit_err[i];
    }
}

void bp_for_midunit(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[OUTPUT_UNIT_NO], double result_mid[MID_UNIT_NO], double out_unit_err[OUTPUT_UNIT_NO]) {
    int i, j, k;
    double unit_err[MID_UNIT_NO];
    double sum_wuerr = 0;

    // 中間層の学習(MID -> INPUT)
    for (i = 0; i < MID_UNIT_NO; i++) {
        sum_wuerr = 0;
        for (j = 0; j < OUTPUT_UNIT_NO; j++) {
            sum_wuerr += w_out[j][i] * out_unit_err[j];
        }
        unit_err[i] = tanhdash(result_mid[i]) * sum_wuerr;

        for (j = 0; j < INPUT_UNIT_NO; j++) {
            w_mid[i][j] -= NNALPHA * input[j] * unit_err[i];
        }
        w_mid[i][j] -= NNALPHA * (-1.0) * unit_err[i];
    }
}

double errsum(double Q[OUTPUT_UNIT_NO], double t[OUTPUT_UNIT_NO]) {
    int i;
    double err = 0.0;

    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        err += (Q[i] - t[i]) * (Q[i] - t[i]);
    }

    return (err);

}

double errcalc(int s[], int a, double r, int s_next[], int a_next, double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], int input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]) {
    double Q, Qnext;
    double result[OUTPUT_UNIT_NO];
    double t;
    double err;
    
    getQ(s, result, w_mid, w_out, input, result_mid);
    Q = result[a];

    getQ(s_next, result, w_mid, w_out, input, result_mid);
    Qnext = result[a_next];

    // 教師データ
    t = updateQvalue(Q, Qnext, r);;
    err = (Q - t) * (Q - t) / 2;

    return err;
}