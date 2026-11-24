#define IDX(a, r) (((a+r-1)*(a+r))/2-a)

int g_oracle_warp_count[50][10][2] = { 0 };      // [layer-1][IDX(res_num, vit_num)][RES|VIT]

enum MODEL {
    RES,
    VIT
};

void init_oracle_warp_count() {
    g_oracle_warp_count[0][IDX(1,1)][0]=27;	g_oracle_warp_count[0][IDX(1,2)][0]=23;	g_oracle_warp_count[0][IDX(1,3)][0]=20;	g_oracle_warp_count[0][IDX(1,4)][0]=20;	g_oracle_warp_count[0][IDX(2,1)][0]=27;	g_oracle_warp_count[0][IDX(2,2)][0]=23;	g_oracle_warp_count[0][IDX(2,3)][0]=22;	g_oracle_warp_count[0][IDX(3,1)][0]=27;	g_oracle_warp_count[0][IDX(3,2)][0]=26;	g_oracle_warp_count[0][IDX(4,1)][0]=28;
    g_oracle_warp_count[1][IDX(1,1)][0]=2;	g_oracle_warp_count[1][IDX(1,2)][0]=2;	g_oracle_warp_count[1][IDX(1,3)][0]=2;	g_oracle_warp_count[1][IDX(1,4)][0]=2;	g_oracle_warp_count[1][IDX(2,1)][0]=2;	g_oracle_warp_count[1][IDX(2,2)][0]=2;	g_oracle_warp_count[1][IDX(2,3)][0]=2;	g_oracle_warp_count[1][IDX(3,1)][0]=3;	g_oracle_warp_count[1][IDX(3,2)][0]=2;	g_oracle_warp_count[1][IDX(4,1)][0]=3;
    g_oracle_warp_count[2][IDX(1,1)][0]=24;	g_oracle_warp_count[2][IDX(1,2)][0]=22;	g_oracle_warp_count[2][IDX(1,3)][0]=21;	g_oracle_warp_count[2][IDX(1,4)][0]=18;	g_oracle_warp_count[2][IDX(2,1)][0]=28;	g_oracle_warp_count[2][IDX(2,2)][0]=24;	g_oracle_warp_count[2][IDX(2,3)][0]=22;	g_oracle_warp_count[2][IDX(3,1)][0]=28;	g_oracle_warp_count[2][IDX(3,2)][0]=26;	g_oracle_warp_count[2][IDX(4,1)][0]=30;
    g_oracle_warp_count[3][IDX(1,1)][0]=5;	g_oracle_warp_count[3][IDX(1,2)][0]=4;	g_oracle_warp_count[3][IDX(1,3)][0]=4;	g_oracle_warp_count[3][IDX(1,4)][0]=3;	g_oracle_warp_count[3][IDX(2,1)][0]=5;	g_oracle_warp_count[3][IDX(2,2)][0]=5;	g_oracle_warp_count[3][IDX(2,3)][0]=4;	g_oracle_warp_count[3][IDX(3,1)][0]=7;	g_oracle_warp_count[3][IDX(3,2)][0]=5;	g_oracle_warp_count[3][IDX(4,1)][0]=6;
    g_oracle_warp_count[4][IDX(1,1)][0]=5;	g_oracle_warp_count[4][IDX(1,2)][0]=5;	g_oracle_warp_count[4][IDX(1,3)][0]=5;	g_oracle_warp_count[4][IDX(1,4)][0]=5;	g_oracle_warp_count[4][IDX(2,1)][0]=5;	g_oracle_warp_count[4][IDX(2,2)][0]=5;	g_oracle_warp_count[4][IDX(2,3)][0]=5;	g_oracle_warp_count[4][IDX(3,1)][0]=6;	g_oracle_warp_count[4][IDX(3,2)][0]=5;	g_oracle_warp_count[4][IDX(4,1)][0]=7;
    g_oracle_warp_count[5][IDX(1,1)][0]=12;	g_oracle_warp_count[5][IDX(1,2)][0]=11;	g_oracle_warp_count[5][IDX(1,3)][0]=10;	g_oracle_warp_count[5][IDX(1,4)][0]=8;	g_oracle_warp_count[5][IDX(2,1)][0]=14;	g_oracle_warp_count[5][IDX(2,2)][0]=12;	g_oracle_warp_count[5][IDX(2,3)][0]=11;	g_oracle_warp_count[5][IDX(3,1)][0]=16;	g_oracle_warp_count[5][IDX(3,2)][0]=14;	g_oracle_warp_count[5][IDX(4,1)][0]=17;
    g_oracle_warp_count[6][IDX(1,1)][0]=16;	g_oracle_warp_count[6][IDX(1,2)][0]=11;	g_oracle_warp_count[6][IDX(1,3)][0]=10;	g_oracle_warp_count[6][IDX(1,4)][0]=7;	g_oracle_warp_count[6][IDX(2,1)][0]=19;	g_oracle_warp_count[6][IDX(2,2)][0]=14;	g_oracle_warp_count[6][IDX(2,3)][0]=14;	g_oracle_warp_count[6][IDX(3,1)][0]=22;	g_oracle_warp_count[6][IDX(3,2)][0]=17;	g_oracle_warp_count[6][IDX(4,1)][0]=23;
    g_oracle_warp_count[7][IDX(1,1)][0]=5;	g_oracle_warp_count[7][IDX(1,2)][0]=5;	g_oracle_warp_count[7][IDX(1,3)][0]=4;	g_oracle_warp_count[7][IDX(1,4)][0]=3;	g_oracle_warp_count[7][IDX(2,1)][0]=6;	g_oracle_warp_count[7][IDX(2,2)][0]=5;	g_oracle_warp_count[7][IDX(2,3)][0]=5;	g_oracle_warp_count[7][IDX(3,1)][0]=7;	g_oracle_warp_count[7][IDX(3,2)][0]=5;	g_oracle_warp_count[7][IDX(4,1)][0]=7;
    g_oracle_warp_count[8][IDX(1,1)][0]=10;	g_oracle_warp_count[8][IDX(1,2)][0]=8;	g_oracle_warp_count[8][IDX(1,3)][0]=8;	g_oracle_warp_count[8][IDX(1,4)][0]=8;	g_oracle_warp_count[8][IDX(2,1)][0]=11;	g_oracle_warp_count[8][IDX(2,2)][0]=11;	g_oracle_warp_count[8][IDX(2,3)][0]=8;	g_oracle_warp_count[8][IDX(3,1)][0]=13;	g_oracle_warp_count[8][IDX(3,2)][0]=11;	g_oracle_warp_count[8][IDX(4,1)][0]=15;
    g_oracle_warp_count[9][IDX(1,1)][0]=4;	g_oracle_warp_count[9][IDX(1,2)][0]=5;	g_oracle_warp_count[9][IDX(1,3)][0]=5;	g_oracle_warp_count[9][IDX(1,4)][0]=4;	g_oracle_warp_count[9][IDX(2,1)][0]=4;	g_oracle_warp_count[9][IDX(2,2)][0]=5;	g_oracle_warp_count[9][IDX(2,3)][0]=5;	g_oracle_warp_count[9][IDX(3,1)][0]=5;	g_oracle_warp_count[9][IDX(3,2)][0]=5;	g_oracle_warp_count[9][IDX(4,1)][0]=5;
    g_oracle_warp_count[10][IDX(1,1)][0]=9;	g_oracle_warp_count[10][IDX(1,2)][0]=7;	g_oracle_warp_count[10][IDX(1,3)][0]=7;	g_oracle_warp_count[10][IDX(1,4)][0]=4;	g_oracle_warp_count[10][IDX(2,1)][0]=12;	g_oracle_warp_count[10][IDX(2,2)][0]=10;	g_oracle_warp_count[10][IDX(2,3)][0]=7;	g_oracle_warp_count[10][IDX(3,1)][0]=13;	g_oracle_warp_count[10][IDX(3,2)][0]=11;	g_oracle_warp_count[10][IDX(4,1)][0]=15;
    g_oracle_warp_count[11][IDX(1,1)][0]=10;	g_oracle_warp_count[11][IDX(1,2)][0]=8;	g_oracle_warp_count[11][IDX(1,3)][0]=7;	g_oracle_warp_count[11][IDX(1,4)][0]=7;	g_oracle_warp_count[11][IDX(2,1)][0]=11;	g_oracle_warp_count[11][IDX(2,2)][0]=10;	g_oracle_warp_count[11][IDX(2,3)][0]=9;	g_oracle_warp_count[11][IDX(3,1)][0]=13;	g_oracle_warp_count[11][IDX(3,2)][0]=11;	g_oracle_warp_count[11][IDX(4,1)][0]=14;
    g_oracle_warp_count[12][IDX(1,1)][0]=2;	g_oracle_warp_count[12][IDX(1,2)][0]=2;	g_oracle_warp_count[12][IDX(1,3)][0]=2;	g_oracle_warp_count[12][IDX(1,4)][0]=2;	g_oracle_warp_count[12][IDX(2,1)][0]=3;	g_oracle_warp_count[12][IDX(2,2)][0]=3;	g_oracle_warp_count[12][IDX(2,3)][0]=5;	g_oracle_warp_count[12][IDX(3,1)][0]=4;	g_oracle_warp_count[12][IDX(3,2)][0]=5;	g_oracle_warp_count[12][IDX(4,1)][0]=5;
    g_oracle_warp_count[13][IDX(1,1)][0]=6;	g_oracle_warp_count[13][IDX(1,2)][0]=5;	g_oracle_warp_count[13][IDX(1,3)][0]=5;	g_oracle_warp_count[13][IDX(1,4)][0]=4;	g_oracle_warp_count[13][IDX(2,1)][0]=7;	g_oracle_warp_count[13][IDX(2,2)][0]=6;	g_oracle_warp_count[13][IDX(2,3)][0]=5;	g_oracle_warp_count[13][IDX(3,1)][0]=8;	g_oracle_warp_count[13][IDX(3,2)][0]=7;	g_oracle_warp_count[13][IDX(4,1)][0]=10;
    g_oracle_warp_count[14][IDX(1,1)][0]=30;	g_oracle_warp_count[14][IDX(1,2)][0]=25;	g_oracle_warp_count[14][IDX(1,3)][0]=22;	g_oracle_warp_count[14][IDX(1,4)][0]=20;	g_oracle_warp_count[14][IDX(2,1)][0]=33;	g_oracle_warp_count[14][IDX(2,2)][0]=28;	g_oracle_warp_count[14][IDX(2,3)][0]=22;	g_oracle_warp_count[14][IDX(3,1)][0]=31;	g_oracle_warp_count[14][IDX(3,2)][0]=31;	g_oracle_warp_count[14][IDX(4,1)][0]=36;
    g_oracle_warp_count[15][IDX(1,1)][0]=2;	g_oracle_warp_count[15][IDX(1,2)][0]=2;	g_oracle_warp_count[15][IDX(1,3)][0]=2;	g_oracle_warp_count[15][IDX(1,4)][0]=2;	g_oracle_warp_count[15][IDX(2,1)][0]=4;	g_oracle_warp_count[15][IDX(2,2)][0]=3;	g_oracle_warp_count[15][IDX(2,3)][0]=4;	g_oracle_warp_count[15][IDX(3,1)][0]=4;	g_oracle_warp_count[15][IDX(3,2)][0]=4;	g_oracle_warp_count[15][IDX(4,1)][0]=5;
    g_oracle_warp_count[16][IDX(1,1)][0]=5;	g_oracle_warp_count[16][IDX(1,2)][0]=5;	g_oracle_warp_count[16][IDX(1,3)][0]=2;	g_oracle_warp_count[16][IDX(1,4)][0]=2;	g_oracle_warp_count[16][IDX(2,1)][0]=5;	g_oracle_warp_count[16][IDX(2,2)][0]=3;	g_oracle_warp_count[16][IDX(2,3)][0]=5;	g_oracle_warp_count[16][IDX(3,1)][0]=4;	g_oracle_warp_count[16][IDX(3,2)][0]=5;	g_oracle_warp_count[16][IDX(4,1)][0]=5;
    g_oracle_warp_count[17][IDX(1,1)][0]=12;	g_oracle_warp_count[17][IDX(1,2)][0]=11;	g_oracle_warp_count[17][IDX(1,3)][0]=9;	g_oracle_warp_count[17][IDX(1,4)][0]=8;	g_oracle_warp_count[17][IDX(2,1)][0]=14;	g_oracle_warp_count[17][IDX(2,2)][0]=12;	g_oracle_warp_count[17][IDX(2,3)][0]=11;	g_oracle_warp_count[17][IDX(3,1)][0]=16;	g_oracle_warp_count[17][IDX(3,2)][0]=14;	g_oracle_warp_count[17][IDX(4,1)][0]=17;
    g_oracle_warp_count[18][IDX(1,1)][0]=16;	g_oracle_warp_count[18][IDX(1,2)][0]=11;	g_oracle_warp_count[18][IDX(1,3)][0]=10;	g_oracle_warp_count[18][IDX(1,4)][0]=7;	g_oracle_warp_count[18][IDX(2,1)][0]=19;	g_oracle_warp_count[18][IDX(2,2)][0]=16;	g_oracle_warp_count[18][IDX(2,3)][0]=11;	g_oracle_warp_count[18][IDX(3,1)][0]=22;	g_oracle_warp_count[18][IDX(3,2)][0]=17;	g_oracle_warp_count[18][IDX(4,1)][0]=25;
    g_oracle_warp_count[19][IDX(1,1)][0]=5;	g_oracle_warp_count[19][IDX(1,2)][0]=4;	g_oracle_warp_count[19][IDX(1,3)][0]=4;	g_oracle_warp_count[19][IDX(1,4)][0]=3;	g_oracle_warp_count[19][IDX(2,1)][0]=6;	g_oracle_warp_count[19][IDX(2,2)][0]=5;	g_oracle_warp_count[19][IDX(2,3)][0]=4;	g_oracle_warp_count[19][IDX(3,1)][0]=7;	g_oracle_warp_count[19][IDX(3,2)][0]=6;	g_oracle_warp_count[19][IDX(4,1)][0]=8;
    g_oracle_warp_count[20][IDX(1,1)][0]=10;	g_oracle_warp_count[20][IDX(1,2)][0]=8;	g_oracle_warp_count[20][IDX(1,3)][0]=8;	g_oracle_warp_count[20][IDX(1,4)][0]=8;	g_oracle_warp_count[20][IDX(2,1)][0]=11;	g_oracle_warp_count[20][IDX(2,2)][0]=10;	g_oracle_warp_count[20][IDX(2,3)][0]=8;	g_oracle_warp_count[20][IDX(3,1)][0]=13;	g_oracle_warp_count[20][IDX(3,2)][0]=11;	g_oracle_warp_count[20][IDX(4,1)][0]=14;
    g_oracle_warp_count[21][IDX(1,1)][0]=4;	g_oracle_warp_count[21][IDX(1,2)][0]=4;	g_oracle_warp_count[21][IDX(1,3)][0]=2;	g_oracle_warp_count[21][IDX(1,4)][0]=2;	g_oracle_warp_count[21][IDX(2,1)][0]=4;	g_oracle_warp_count[21][IDX(2,2)][0]=4;	g_oracle_warp_count[21][IDX(2,3)][0]=4;	g_oracle_warp_count[21][IDX(3,1)][0]=5;	g_oracle_warp_count[21][IDX(3,2)][0]=5;	g_oracle_warp_count[21][IDX(4,1)][0]=7;
    g_oracle_warp_count[22][IDX(1,1)][0]=9;	g_oracle_warp_count[22][IDX(1,2)][0]=7;	g_oracle_warp_count[22][IDX(1,3)][0]=7;	g_oracle_warp_count[22][IDX(1,4)][0]=4;	g_oracle_warp_count[22][IDX(2,1)][0]=13;	g_oracle_warp_count[22][IDX(2,2)][0]=10;	g_oracle_warp_count[22][IDX(2,3)][0]=7;	g_oracle_warp_count[22][IDX(3,1)][0]=16;	g_oracle_warp_count[22][IDX(3,2)][0]=14;	g_oracle_warp_count[22][IDX(4,1)][0]=19;
    g_oracle_warp_count[23][IDX(1,1)][0]=10;	g_oracle_warp_count[23][IDX(1,2)][0]=9;	g_oracle_warp_count[23][IDX(1,3)][0]=7;	g_oracle_warp_count[23][IDX(1,4)][0]=7;	g_oracle_warp_count[23][IDX(2,1)][0]=11;	g_oracle_warp_count[23][IDX(2,2)][0]=10;	g_oracle_warp_count[23][IDX(2,3)][0]=9;	g_oracle_warp_count[23][IDX(3,1)][0]=13;	g_oracle_warp_count[23][IDX(3,2)][0]=11;	g_oracle_warp_count[23][IDX(4,1)][0]=14;
    g_oracle_warp_count[24][IDX(1,1)][0]=2;	g_oracle_warp_count[24][IDX(1,2)][0]=2;	g_oracle_warp_count[24][IDX(1,3)][0]=2;	g_oracle_warp_count[24][IDX(1,4)][0]=2;	g_oracle_warp_count[24][IDX(2,1)][0]=4;	g_oracle_warp_count[24][IDX(2,2)][0]=3;	g_oracle_warp_count[24][IDX(2,3)][0]=3;	g_oracle_warp_count[24][IDX(3,1)][0]=5;	g_oracle_warp_count[24][IDX(3,2)][0]=5;	g_oracle_warp_count[24][IDX(4,1)][0]=5;
    g_oracle_warp_count[25][IDX(1,1)][0]=6;	g_oracle_warp_count[25][IDX(1,2)][0]=5;	g_oracle_warp_count[25][IDX(1,3)][0]=4;	g_oracle_warp_count[25][IDX(1,4)][0]=3;	g_oracle_warp_count[25][IDX(2,1)][0]=7;	g_oracle_warp_count[25][IDX(2,2)][0]=6;	g_oracle_warp_count[25][IDX(2,3)][0]=6;	g_oracle_warp_count[25][IDX(3,1)][0]=9;	g_oracle_warp_count[25][IDX(3,2)][0]=7;	g_oracle_warp_count[25][IDX(4,1)][0]=10;
    g_oracle_warp_count[26][IDX(1,1)][0]=28;	g_oracle_warp_count[26][IDX(1,2)][0]=28;	g_oracle_warp_count[26][IDX(1,3)][0]=25;	g_oracle_warp_count[26][IDX(1,4)][0]=20;	g_oracle_warp_count[26][IDX(2,1)][0]=33;	g_oracle_warp_count[26][IDX(2,2)][0]=28;	g_oracle_warp_count[26][IDX(2,3)][0]=28;	g_oracle_warp_count[26][IDX(3,1)][0]=28;	g_oracle_warp_count[26][IDX(3,2)][0]=28;	g_oracle_warp_count[26][IDX(4,1)][0]=33;
    g_oracle_warp_count[27][IDX(1,1)][0]=2;	g_oracle_warp_count[27][IDX(1,2)][0]=2;	g_oracle_warp_count[27][IDX(1,3)][0]=2;	g_oracle_warp_count[27][IDX(1,4)][0]=2;	g_oracle_warp_count[27][IDX(2,1)][0]=4;	g_oracle_warp_count[27][IDX(2,2)][0]=3;	g_oracle_warp_count[27][IDX(2,3)][0]=3;	g_oracle_warp_count[27][IDX(3,1)][0]=4;	g_oracle_warp_count[27][IDX(3,2)][0]=4;	g_oracle_warp_count[27][IDX(4,1)][0]=8;
    g_oracle_warp_count[28][IDX(1,1)][0]=2;	g_oracle_warp_count[28][IDX(1,2)][0]=2;	g_oracle_warp_count[28][IDX(1,3)][0]=2;	g_oracle_warp_count[28][IDX(1,4)][0]=2;	g_oracle_warp_count[28][IDX(2,1)][0]=4;	g_oracle_warp_count[28][IDX(2,2)][0]=3;	g_oracle_warp_count[28][IDX(2,3)][0]=3;	g_oracle_warp_count[28][IDX(3,1)][0]=8;	g_oracle_warp_count[28][IDX(3,2)][0]=5;	g_oracle_warp_count[28][IDX(4,1)][0]=8;
    g_oracle_warp_count[29][IDX(1,1)][0]=12;	g_oracle_warp_count[29][IDX(1,2)][0]=11;	g_oracle_warp_count[29][IDX(1,3)][0]=9;	g_oracle_warp_count[29][IDX(1,4)][0]=8;	g_oracle_warp_count[29][IDX(2,1)][0]=14;	g_oracle_warp_count[29][IDX(2,2)][0]=12;	g_oracle_warp_count[29][IDX(2,3)][0]=11;	g_oracle_warp_count[29][IDX(3,1)][0]=16;	g_oracle_warp_count[29][IDX(3,2)][0]=14;	g_oracle_warp_count[29][IDX(4,1)][0]=17;
    g_oracle_warp_count[30][IDX(1,1)][0]=16;	g_oracle_warp_count[30][IDX(1,2)][0]=11;	g_oracle_warp_count[30][IDX(1,3)][0]=10;	g_oracle_warp_count[30][IDX(1,4)][0]=7;	g_oracle_warp_count[30][IDX(2,1)][0]=20;	g_oracle_warp_count[30][IDX(2,2)][0]=16;	g_oracle_warp_count[30][IDX(2,3)][0]=14;	g_oracle_warp_count[30][IDX(3,1)][0]=25;	g_oracle_warp_count[30][IDX(3,2)][0]=19;	g_oracle_warp_count[30][IDX(4,1)][0]=28;
    g_oracle_warp_count[31][IDX(1,1)][0]=5;	g_oracle_warp_count[31][IDX(1,2)][0]=4;	g_oracle_warp_count[31][IDX(1,3)][0]=4;	g_oracle_warp_count[31][IDX(1,4)][0]=3;	g_oracle_warp_count[31][IDX(2,1)][0]=6;	g_oracle_warp_count[31][IDX(2,2)][0]=5;	g_oracle_warp_count[31][IDX(2,3)][0]=5;	g_oracle_warp_count[31][IDX(3,1)][0]=7;	g_oracle_warp_count[31][IDX(3,2)][0]=6;	g_oracle_warp_count[31][IDX(4,1)][0]=8;
    g_oracle_warp_count[32][IDX(1,1)][0]=10;	g_oracle_warp_count[32][IDX(1,2)][0]=8;	g_oracle_warp_count[32][IDX(1,3)][0]=8;	g_oracle_warp_count[32][IDX(1,4)][0]=8;	g_oracle_warp_count[32][IDX(2,1)][0]=11;	g_oracle_warp_count[32][IDX(2,2)][0]=11;	g_oracle_warp_count[32][IDX(2,3)][0]=8;	g_oracle_warp_count[32][IDX(3,1)][0]=13;	g_oracle_warp_count[32][IDX(3,2)][0]=11;	g_oracle_warp_count[32][IDX(4,1)][0]=14;
    g_oracle_warp_count[33][IDX(1,1)][0]=3;	g_oracle_warp_count[33][IDX(1,2)][0]=3;	g_oracle_warp_count[33][IDX(1,3)][0]=3;	g_oracle_warp_count[33][IDX(1,4)][0]=2;	g_oracle_warp_count[33][IDX(2,1)][0]=4;	g_oracle_warp_count[33][IDX(2,2)][0]=4;	g_oracle_warp_count[33][IDX(2,3)][0]=4;	g_oracle_warp_count[33][IDX(3,1)][0]=6;	g_oracle_warp_count[33][IDX(3,2)][0]=5;	g_oracle_warp_count[33][IDX(4,1)][0]=7;
    g_oracle_warp_count[34][IDX(1,1)][0]=16;	g_oracle_warp_count[34][IDX(1,2)][0]=11;	g_oracle_warp_count[34][IDX(1,3)][0]=10;	g_oracle_warp_count[34][IDX(1,4)][0]=7;	g_oracle_warp_count[34][IDX(2,1)][0]=22;	g_oracle_warp_count[34][IDX(2,2)][0]=16;	g_oracle_warp_count[34][IDX(2,3)][0]=14;	g_oracle_warp_count[34][IDX(3,1)][0]=24;	g_oracle_warp_count[34][IDX(3,2)][0]=19;	g_oracle_warp_count[34][IDX(4,1)][0]=28;
    g_oracle_warp_count[35][IDX(1,1)][0]=10;	g_oracle_warp_count[35][IDX(1,2)][0]=9;	g_oracle_warp_count[35][IDX(1,3)][0]=7;	g_oracle_warp_count[35][IDX(1,4)][0]=7;	g_oracle_warp_count[35][IDX(2,1)][0]=11;	g_oracle_warp_count[35][IDX(2,2)][0]=10;	g_oracle_warp_count[35][IDX(2,3)][0]=9;	g_oracle_warp_count[35][IDX(3,1)][0]=13;	g_oracle_warp_count[35][IDX(3,2)][0]=11;	g_oracle_warp_count[35][IDX(4,1)][0]=14;
    g_oracle_warp_count[36][IDX(1,1)][0]=2;	g_oracle_warp_count[36][IDX(1,2)][0]=2;	g_oracle_warp_count[36][IDX(1,3)][0]=2;	g_oracle_warp_count[36][IDX(1,4)][0]=2;	g_oracle_warp_count[36][IDX(2,1)][0]=5;	g_oracle_warp_count[36][IDX(2,2)][0]=3;	g_oracle_warp_count[36][IDX(2,3)][0]=3;	g_oracle_warp_count[36][IDX(3,1)][0]=5;	g_oracle_warp_count[36][IDX(3,2)][0]=5;	g_oracle_warp_count[36][IDX(4,1)][0]=5;
    g_oracle_warp_count[37][IDX(1,1)][0]=6;	g_oracle_warp_count[37][IDX(1,2)][0]=5;	g_oracle_warp_count[37][IDX(1,3)][0]=5;	g_oracle_warp_count[37][IDX(1,4)][0]=4;	g_oracle_warp_count[37][IDX(2,1)][0]=7;	g_oracle_warp_count[37][IDX(2,2)][0]=6;	g_oracle_warp_count[37][IDX(2,3)][0]=6;	g_oracle_warp_count[37][IDX(3,1)][0]=9;	g_oracle_warp_count[37][IDX(3,2)][0]=7;	g_oracle_warp_count[37][IDX(4,1)][0]=10;
    g_oracle_warp_count[38][IDX(1,1)][0]=28;	g_oracle_warp_count[38][IDX(1,2)][0]=28;	g_oracle_warp_count[38][IDX(1,3)][0]=22;	g_oracle_warp_count[38][IDX(1,4)][0]=20;	g_oracle_warp_count[38][IDX(2,1)][0]=34;	g_oracle_warp_count[38][IDX(2,2)][0]=28;	g_oracle_warp_count[38][IDX(2,3)][0]=25;	g_oracle_warp_count[38][IDX(3,1)][0]=33;	g_oracle_warp_count[38][IDX(3,2)][0]=28;	g_oracle_warp_count[38][IDX(4,1)][0]=33;
    g_oracle_warp_count[39][IDX(1,1)][0]=2;	g_oracle_warp_count[39][IDX(1,2)][0]=3;	g_oracle_warp_count[39][IDX(1,3)][0]=2;	g_oracle_warp_count[39][IDX(1,4)][0]=2;	g_oracle_warp_count[39][IDX(2,1)][0]=4;	g_oracle_warp_count[39][IDX(2,2)][0]=3;	g_oracle_warp_count[39][IDX(2,3)][0]=3;	g_oracle_warp_count[39][IDX(3,1)][0]=4;	g_oracle_warp_count[39][IDX(3,2)][0]=4;	g_oracle_warp_count[39][IDX(4,1)][0]=5;
    g_oracle_warp_count[40][IDX(1,1)][0]=2;	g_oracle_warp_count[40][IDX(1,2)][0]=1;	g_oracle_warp_count[40][IDX(1,3)][0]=1;	g_oracle_warp_count[40][IDX(1,4)][0]=1;	g_oracle_warp_count[40][IDX(2,1)][0]=2;	g_oracle_warp_count[40][IDX(2,2)][0]=2;	g_oracle_warp_count[40][IDX(2,3)][0]=2;	g_oracle_warp_count[40][IDX(3,1)][0]=4;	g_oracle_warp_count[40][IDX(3,2)][0]=2;	g_oracle_warp_count[40][IDX(4,1)][0]=4;
    g_oracle_warp_count[41][IDX(1,1)][0]=13;	g_oracle_warp_count[41][IDX(1,2)][0]=11;	g_oracle_warp_count[41][IDX(1,3)][0]=9;	g_oracle_warp_count[41][IDX(1,4)][0]=9;	g_oracle_warp_count[41][IDX(2,1)][0]=14;	g_oracle_warp_count[41][IDX(2,2)][0]=13;	g_oracle_warp_count[41][IDX(2,3)][0]=11;	g_oracle_warp_count[41][IDX(3,1)][0]=17;	g_oracle_warp_count[41][IDX(3,2)][0]=14;	g_oracle_warp_count[41][IDX(4,1)][0]=17;
    g_oracle_warp_count[42][IDX(1,1)][0]=16;	g_oracle_warp_count[42][IDX(1,2)][0]=11;	g_oracle_warp_count[42][IDX(1,3)][0]=10;	g_oracle_warp_count[42][IDX(1,4)][0]=10;	g_oracle_warp_count[42][IDX(2,1)][0]=19;	g_oracle_warp_count[42][IDX(2,2)][0]=16;	g_oracle_warp_count[42][IDX(2,3)][0]=14;	g_oracle_warp_count[42][IDX(3,1)][0]=24;	g_oracle_warp_count[42][IDX(3,2)][0]=19;	g_oracle_warp_count[42][IDX(4,1)][0]=25;
    g_oracle_warp_count[43][IDX(1,1)][0]=5;	g_oracle_warp_count[43][IDX(1,2)][0]=4;	g_oracle_warp_count[43][IDX(1,3)][0]=4;	g_oracle_warp_count[43][IDX(1,4)][0]=3;	g_oracle_warp_count[43][IDX(2,1)][0]=6;	g_oracle_warp_count[43][IDX(2,2)][0]=5;	g_oracle_warp_count[43][IDX(2,3)][0]=5;	g_oracle_warp_count[43][IDX(3,1)][0]=7;	g_oracle_warp_count[43][IDX(3,2)][0]=6;	g_oracle_warp_count[43][IDX(4,1)][0]=8;
    g_oracle_warp_count[44][IDX(1,1)][0]=11;	g_oracle_warp_count[44][IDX(1,2)][0]=8;	g_oracle_warp_count[44][IDX(1,3)][0]=8;	g_oracle_warp_count[44][IDX(1,4)][0]=8;	g_oracle_warp_count[44][IDX(2,1)][0]=11;	g_oracle_warp_count[44][IDX(2,2)][0]=11;	g_oracle_warp_count[44][IDX(2,3)][0]=11;	g_oracle_warp_count[44][IDX(3,1)][0]=13;	g_oracle_warp_count[44][IDX(3,2)][0]=11;	g_oracle_warp_count[44][IDX(4,1)][0]=15;
    g_oracle_warp_count[45][IDX(1,1)][0]=3;	g_oracle_warp_count[45][IDX(1,2)][0]=3;	g_oracle_warp_count[45][IDX(1,3)][0]=3;	g_oracle_warp_count[45][IDX(1,4)][0]=3;	g_oracle_warp_count[45][IDX(2,1)][0]=7;	g_oracle_warp_count[45][IDX(2,2)][0]=5;	g_oracle_warp_count[45][IDX(2,3)][0]=4;	g_oracle_warp_count[45][IDX(3,1)][0]=8;	g_oracle_warp_count[45][IDX(3,2)][0]=7;	g_oracle_warp_count[45][IDX(4,1)][0]=9;
    g_oracle_warp_count[46][IDX(1,1)][0]=20;	g_oracle_warp_count[46][IDX(1,2)][0]=17;	g_oracle_warp_count[46][IDX(1,3)][0]=14;	g_oracle_warp_count[46][IDX(1,4)][0]=10;	g_oracle_warp_count[46][IDX(2,1)][0]=25;	g_oracle_warp_count[46][IDX(2,2)][0]=20;	g_oracle_warp_count[46][IDX(2,3)][0]=17;	g_oracle_warp_count[46][IDX(3,1)][0]=25;	g_oracle_warp_count[46][IDX(3,2)][0]=25;	g_oracle_warp_count[46][IDX(4,1)][0]=25;
    g_oracle_warp_count[47][IDX(1,1)][0]=10;	g_oracle_warp_count[47][IDX(1,2)][0]=9;	g_oracle_warp_count[47][IDX(1,3)][0]=7;	g_oracle_warp_count[47][IDX(1,4)][0]=7;	g_oracle_warp_count[47][IDX(2,1)][0]=11;	g_oracle_warp_count[47][IDX(2,2)][0]=10;	g_oracle_warp_count[47][IDX(2,3)][0]=9;	g_oracle_warp_count[47][IDX(3,1)][0]=13;	g_oracle_warp_count[47][IDX(3,2)][0]=11;	g_oracle_warp_count[47][IDX(4,1)][0]=14;
    g_oracle_warp_count[48][IDX(1,1)][0]=4;	g_oracle_warp_count[48][IDX(1,2)][0]=2;	g_oracle_warp_count[48][IDX(1,3)][0]=2;	g_oracle_warp_count[48][IDX(1,4)][0]=2;	g_oracle_warp_count[48][IDX(2,1)][0]=4;	g_oracle_warp_count[48][IDX(2,2)][0]=3;	g_oracle_warp_count[48][IDX(2,3)][0]=3;	g_oracle_warp_count[48][IDX(3,1)][0]=5;	g_oracle_warp_count[48][IDX(3,2)][0]=5;	g_oracle_warp_count[48][IDX(4,1)][0]=5;
    g_oracle_warp_count[49][IDX(1,1)][0]=19;	g_oracle_warp_count[49][IDX(1,2)][0]=22;	g_oracle_warp_count[49][IDX(1,3)][0]=15;	g_oracle_warp_count[49][IDX(1,4)][0]=16;	g_oracle_warp_count[49][IDX(2,1)][0]=4;	g_oracle_warp_count[49][IDX(2,2)][0]=4;	g_oracle_warp_count[49][IDX(2,3)][0]=4;	g_oracle_warp_count[49][IDX(3,1)][0]=4;	g_oracle_warp_count[49][IDX(3,2)][0]=4;	g_oracle_warp_count[49][IDX(4,1)][0]=4;

}

void static_allocation(
    int* r_MKN, int r_merge,
    int* v_MKN, int v_merge,
    int warpPerSM, int layer, int* r_warp, int* v_warp)
{
    // 실행 횟수가 최소가 되는 개수 찾기
    int optimal_res_warp_start = 0;
    int optimal_res_warp_end = 0;
    float min_ops = 1e+37;

    // SM 당 warp 개수를 바꿔가며 연산 횟수를 계산하고, 최소가 되는 할당 개수 찾기
    for (int i = 1; i < warpPerSM; i++) {
        int r_threadsPerSM = WARP_SIZE * i;
        int v_threadsPerSM = WARP_SIZE * (warpPerSM - i);

        float r_ops = ceil((float)(r_MKN[0]*r_MKN[2]) / (SM_COUNT * r_threadsPerSM)) * r_MKN[1] * (r_merge + 1);
        float v_ops = ceil((float)(v_MKN[0]*v_MKN[2]) / (SM_COUNT * v_threadsPerSM)) * v_MKN[1] * (v_merge + 1);

        if (min_ops >= max(r_ops, v_ops)) {
            optimal_res_warp_end = i;
            if (min_ops > max(r_ops, v_ops)) {
                optimal_res_warp_start = i;
            }
            
            min_ops = max(r_ops, v_ops);
        }
    }

    // 최소가 되는 할당 개수를 *r_warp, *v_warp에 저장
    *r_warp = (optimal_res_warp_start + optimal_res_warp_end) / 2;
    *v_warp = warpPerSM - *r_warp;
}

void oracle_allocation(
    int layer, int r_merge, int v_merge, 
    int* r_warp, int* v_warp)
{
    *r_warp = g_oracle_warp_count[layer-1][IDX(r_merge, v_merge)][RES];
    *v_warp = 48 - *r_warp;
}

void warp_allocation(
    int* r_MKN, int r_merge,
    int* v_MKN, int v_merge,
    int warpPerSM, int layer, int* r_warp, int* v_warp,
    int mode) // mode - 0: Merge and Fusion (prior work) / 1: TLP Balancer / 3: Oracle
{
    // 하나의 model만 실행하면 해당 model에 48개 할당
    if (r_merge == 0 || v_merge == 0) {
        if (r_merge == 0) {
            *r_warp = 0;
            *v_warp = warpPerSM;
        }
        else {
            *r_warp = warpPerSM;
            *v_warp = 0;
        }

        return;
    }

    // warm up 할 때는 static하게 결정
    if (r_merge == 5 && v_merge == 5) {
        static_allocation(r_MKN, r_merge,
                          v_MKN, v_merge,
                          warpPerSM, layer, r_warp, v_warp);

        return;
    }

    // 실제 실행
    switch (mode) {
    // Merge and Fusion (prior work): 균등 할당
    case 0:
        *r_warp = 24;
        *v_warp = 24;
        break;

    // TLP Balancer: 연산 횟수 예측으로 할당
    case 1:
        static_allocation(r_MKN, r_merge,
                          v_MKN, v_merge,
                          warpPerSM, layer, r_warp, v_warp);
        break;

    // Oracle: 실측으로 찾은 최적 할당
    case 3:
        oracle_allocation(layer, r_merge, v_merge, r_warp, v_warp);
        break;
    }
}