#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cmath>
using namespace std;

// total number of layers of weights and activations (50 layers' weights and 49 layers' activations)
int num_layers = 99;
// total number of weights + total number of activations
int total_num_data = 32342720;
// filename of rate-distortion curves
string filename_data_points = "resnet_50_rate_distortion_curves.txt";

// structure of one node in rate-distortion curves
// 'length': total bitrate
// 'error': output error
// 'layer': which layer of weights or activations
// 'dead_zone': the size of dead zone
// 'quant': number of quantization centroid
struct Node{
    int layer, dead_zone, quant;
    float length;
    float error;
};

// define rate-distortion curves.
vector<Node> *data_points;

// eps for binary search.
float eps = 1e-10;

// binary search to find optimal slope under Pareto-condition.
void solve(vector<int> &solutions_dead_zone , vector<int> &solutions_quant , float target){
    for(int i = 0 ; i < num_layers ; i ++){
        solutions_dead_zone.push_back(0);
        solutions_quant.push_back(0);
    }

    float left = -100000000.0;
    float right = 0.0;

    while(right - left >= eps){
        float total_size = 0.0;
        float mid = (left + right) * 0.5;

        for(int i = 0 ; i < num_layers ; i ++){
            int selected = -1;
            float intercept = 100000000;

            for(int j = 0 ; j < data_points[i].size() ; j ++){
                float p = data_points[i][j].length;
                float q = data_points[i][j].error;
                float b = q - mid * p;

                if(b < intercept || selected == -1){
                    intercept = b;
                    selected = j;
                }
            }

            solutions_dead_zone[i] = data_points[i][selected].dead_zone;
            solutions_quant[i] = data_points[i][selected].quant;
            total_size += data_points[i][selected].length;
        }

        if(total_size > target) right = mid;
        else left = mid;
    }
}

int main(){
    // define rate-distortion curves
    // totally have 'num_layers' curves
    data_points = new vector<Node>[num_layers];

    // read rate-distortion curves' data from file
    FILE *file_data_points = fopen(filename_data_points.c_str() , "r");
    for(int i = 0 ; i < num_layers ; i ++){
        int m;
        fscanf(file_data_points , "%d" , &m);

        for(int j = 0 ; j < m ; j ++){
            int layer, dead_zone, quant;
            float length;
            float error;

            fscanf(file_data_points , "%d %d %d %f %f\n" , &layer , &dead_zone , &quant , &length , &error);

            Node node;
            node.layer = layer, node.dead_zone = dead_zone, node.quant = quant, node.length = length , node.error = error;

            data_points[i].push_back(node);
        }
    }

    // optimize bit allocation given the average bitrate, from 2.0 bits on average to 16.0 bits on average.
    for(int i = 20 ; i <= 160 ; i ++){
        vector<int> solutions_dead_zone;
        vector<int> solutions_quant;
        solve(solutions_dead_zone , solutions_quant , 1.0 * i / 10.0 * total_num_data);
        char filename[1010];
        sprintf(filename , "bit_allocations_%d.txt" , i);

        // write optimum bit allocation to file
        FILE *out = fopen(filename , "w");
        for(int i = 0 ; i < solutions_dead_zone.size() ; i ++){
            fprintf(out , "%d %d\n" , solutions_dead_zone[i] , solutions_quant[i]);
        }
        fclose(out);

        printf("finish optimization of bit allocation at %.2f bits on average.\n" , 1.0 * i / 10.0);
    }
    fclose(file_data_points);

    return 0;
}
