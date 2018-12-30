#include <stdio.h>
#include <stdlib.h>
#include "MT.h"

#define STATE_NO 273
#define ACTION_NO 4
#define ALPHA 0.1
#define EPSILON 0.3
#define GAMMA 0.3
#define L 1
#define EPISODE_NO 10000
#define STEP_NO 1000
#define START_S 232
#define GORL_S 250
#define ACTION_UP 0
#define ACTION_DOWN 1
#define ACTION_RIGHT 2
#define ACTION_LEFT 3
#define SEED 32767

/*
0 1 2 ........ 18 20
21 22 ........ 40 41 
....................
....................
....................
231 232 .... 250 251
252 ............ 272
*/


void initQ(double Q[STATE_NO][ACTION_NO]) {
    int i, j;
    for (i = 0; i < STATE_NO; i++) {
        for (j = 0; j < ACTION_NO; j++) {
            Q[i][j] = genrand_real1();
        }
    }
}

int pi(int s, double Q[STATE_NO][ACTION_NO]) {
    int i;
    int max_a;

    if (genrand_real1() < EPSILON) {
        return (genrand_int32() % ACTION_NO);
    }
    else {
        max_a = 0;
        for (i = 0; i < ACTION_NO; i++) {
            if (Q[s][i] > Q[s][max_a]) {
                max_a = i;
            }
        }
        return max_a;
    }
}

int statetransition(int s, int a) {
    int s_next;
    int nexts[] = {-21, 21, 1, -1};

    s_next = s + nexts[a];

    // 遷移結果が壁の場合
    if ((s_next <= 20) || (s_next >= 252) || ((s_next % 21 == 0) || (s_next % 21 == 20))) {
        s_next = s;
    }
    return s_next;
}

int reword(int s, int s_next) {
    int r;
    if (s_next == s) {
        r = -100;
    }
    else if ((s_next > 232) && (s_next < 250)) {
        r = -100;
    }
    else if (s_next == GORL_S) {
        r = 100;
    }
    else {
        r = -1;
    }

    return r;
}
void SARSAupdateQ(int s, int a, int r, int s_next, int a_next, double Q[STATE_NO][ACTION_NO]) {
    Q[s][a] = Q[s][a] + ALPHA * (r - Q[s][a] + GAMMA * Q[s_next][a_next]);
}

void QLearningupdateQ(int s, int a, int r, int s_next, double Q[STATE_NO][ACTION_NO]) {
    int i;
    int max_a = 0;
    for (i = 0; i < ACTION_NO; i++) {
        if (Q[s_next][i] > Q[s_next][max_a]) {
            max_a = i;
        }
    }
    Q[s][a] = Q[s][a] + ALPHA * (r - Q[s][a] + GAMMA * Q[s_next][max_a]);
}

void printQ(double Q[STATE_NO][ACTION_NO]) {
    int i, j;
    int max_a;
    int count = 0;

    for (i = 0; i < STATE_NO; i++) {
        if (i % 21 == 0) {
            printf("|");
        }
        else if (i % 21 == 20) {
            printf("|\n");
        }
        else if ((i <= 20) || (i >= 231 && i != START_S && i != GORL_S)) {
            printf("- ");
        }
        else {
            max_a = 0;
            for (j = 0; j < ACTION_NO; j++) {
                if (Q[i][j] > Q[i][max_a]) {
                    max_a = j;
                }
            }
            if (max_a == ACTION_UP) {
                printf("↑ ");
            }
            else if (max_a == ACTION_DOWN) {
                printf("↓ ");
            }
            else if (max_a == ACTION_RIGHT) {
                printf("→ ");
            }
            else if (max_a == ACTION_LEFT) {
                printf("← ");
            }
        }
    }
    printf("\n");

    // for (i = 0; i < STATE_NO; i++) {
    //     printf("%d ", count);
    //     count++;
    //     for (j = 0; j < ACTION_NO; j++) {
    //         printf("%lf ", Q[i][j]);
    //     }
    //     printf("\n");
    // }
}

void QLearning() {
    double Q[STATE_NO][ACTION_NO];
    int i, j, k;
    int s, a;
    int s_next, a_next;
    int r;
    
    initQ(Q);
    for (i = 0; i < L; i++) {
        // printQ(Q);
        for (j = 0; j < EPISODE_NO; j++) {
            s = START_S;
            for (k = 0; k < STEP_NO; k++) {
                a = pi(s, Q);
                s_next = statetransition(s, a);
                r = reword(s, s_next);
                QLearningupdateQ(s, a, r, s_next, Q);
                s = s_next;
                if (s == GORL_S) {
                    break;
                }
                else if ((s > 232) && (s < 250)) {
                    s = START_S;
                }
            }
        }
        printQ(Q);
    }
}

void SARSA() {
    double Q[STATE_NO][ACTION_NO];
    int i, j, k;
    int s, a;
    int s_next, a_next;
    int r;
    
    initQ(Q);
    for (i = 0; i < L; i++) {
        // printQ(Q);
        for (j = 0; j < EPISODE_NO; j++) {
            s = START_S;
            for (k = 0; k < STEP_NO; k++) {
                a = pi(s, Q);
                s_next = statetransition(s, a);
                r = reword(s, s_next);
                a_next = pi(s_next, Q);
                SARSAupdateQ(s, a, r, s_next, a_next, Q);
                s = s_next;
                if (s == GORL_S) {
                    break;
                }
                else if ((s > 232) && (s < 250)) {
                    s = START_S;
                }
            }
        }
        printQ(Q);
    }
}

int main(void) {
    int i, j, k;
    int s, a;
    int s_next, a_next;
    int r;

    init_genrand(SEED);

    printf("SARSA\n");
    SARSA();
    printf("Q\n");
    QLearning();
    


    return 0;
}