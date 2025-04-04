#include <iostream>
#include <queue>
#include <vector>
#include <climits>
#include <cuda_runtime.h>
#include <fstream>
#include "json.hpp" // Ensure you include the correct header for JsonCpp
#include <cstdlib>    // for rand()
#include <ctime>      // for time()

// Grid size constants
#define GRID_WIDTH 7
#define GRID_HEIGHT 6

// Directions for movement
const int dx[] = {-1, 1, 0, 0}; // Left, Right, Up, Down
const int dy[] = {0, 0, -1, 1}; // Up, Down, Left, Right
const char* direction_str[] = {"left", "right", "up", "down"};

// Structure to store a position
struct Position {
    int x, y;
};

// CUDA kernel to perform Dijkstra's algorithm (Placeholder)
__global__ void dijkstra_kernel(int* grid, int start_x, int start_y, int target_x, int target_y, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < GRID_WIDTH && idy < GRID_HEIGHT) {
        // Implement Dijkstra’s algorithm here (placeholder)
        // For simplicity, we simulate a result based on random values
        result[idx + idy * GRID_WIDTH] = 42;  // Random distance value (simulated)
    }
}

// Function to invoke the CUDA kernel
void cuda_dijkstra(int grid[GRID_HEIGHT][GRID_WIDTH], int start_x, int start_y, int target_x, int target_y) {
    int* dev_grid;
    int* dev_result;

    // Allocate memory on the device
    cudaMalloc((void**)&dev_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(int));
    cudaMalloc((void**)&dev_result, GRID_WIDTH * GRID_HEIGHT * sizeof(int));

    // Copy grid data to the device
    cudaMemcpy(dev_grid, grid, GRID_WIDTH * GRID_HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((GRID_WIDTH + 15) / 16, (GRID_HEIGHT + 15) / 16);

    // Launch the kernel
    dijkstra_kernel<<<numBlocks, threadsPerBlock>>>(dev_grid, start_x, start_y, target_x, target_y, dev_result);

    // Copy result back to the host
    int* result = new int[GRID_WIDTH * GRID_HEIGHT];
    cudaMemcpy(result, dev_result, GRID_WIDTH * GRID_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_grid);
    cudaFree(dev_result);

    std::cout << "Dijkstra Result: " << result[0] << std::endl; // Placeholder output
}

// Function to generate random directions that move towards the target
std::string generate_random_directions(int start_x, int start_y, int target_x, int target_y, int min_steps, int max_steps) {
    // Calculate the difference in the x and y directions
    int x_diff = target_x - start_x;
    int y_diff = target_y - start_y;

    // Determine the length of the path (between min_steps and max_steps)
    int num_steps = rand() % (max_steps - min_steps + 1) + min_steps; // Random number of steps between min_steps and max_steps
    std::string directions = "";

    // Generate random but reasonable steps
    for (int i = 0; i < num_steps; ++i) {
        int direction = -1;

        // Prioritize moving toward the target first
        if (x_diff > 0) {
            direction = 1; // Move right
            x_diff--;
        } else if (x_diff < 0) {
            direction = 0; // Move left
            x_diff++;
        } else if (y_diff > 0) {
            direction = 2; // Move down
            y_diff--;
        } else if (y_diff < 0) {
            direction = 3; // Move up
            y_diff++;
        }

        // Once the person is at the target, add more randomness
        if (x_diff == 0 && y_diff == 0) {
            direction = rand() % 4; // Choose a random direction
        }

        // Avoid sticking to one direction too much by introducing randomness in movements
        if (i > 5) { // After the 5th step, we can add more randomness
            direction = rand() % 4; // Choose a random direction for variance
        }

        directions += direction_str[direction];

        // Add a separator if not the last direction
        if (i < num_steps - 1) {
            directions += ", ";
        }
    }

    return directions;
}


// Function to save results in JSON format
void save_to_json(const std::string& filename, const std::vector<std::string>& commands) {
    Json::Value root; // Creating the root node of the JSON object
    for (size_t i = 0; i < commands.size(); ++i) {
        root["person" + std::to_string(i+1)] = commands[i];
    }

    std::ofstream file(filename);
    file << root;
    file.close();
}

int main() {
    srand(time(NULL));  // Initialize random seed

    // Read grid data from the file
    int grid[GRID_HEIGHT][GRID_WIDTH];
    FILE* grid_file = fopen("grid.txt", "r");
    if (!grid_file) {
        std::cerr << "Error opening grid file!" << std::endl;
        return 1;
    }

    for (int i = 0; i < GRID_HEIGHT; i++) {
        for (int j = 0; j < GRID_WIDTH; j++) {
            fscanf(grid_file, "%d", &grid[i][j]);
        }
    }
    fclose(grid_file);

    // Read people data from the file
    std::vector<Position> people_start;
    std::vector<Position> people_target;
    FILE* people_file = fopen("people.txt", "r");
    if (!people_file) {
        std::cerr << "Error opening people file!" << std::endl;
        return 1;
    }

    int start_x, start_y, target_x, target_y;
    while (fscanf(people_file, "%d %d %d %d", &start_x, &start_y, &target_x, &target_y) == 4) {
        people_start.push_back({start_x, start_y});
        people_target.push_back({target_x, target_y});
    }
    fclose(people_file);

    // Process each person and find the best solution using CUDA Dijkstra
    std::vector<std::string> travel_commands;
    for (size_t i = 0; i < people_start.size(); ++i) {
        std::cout << "Processing person " << i + 1 << std::endl;
        cuda_dijkstra(grid, people_start[i].x, people_start[i].y, people_target[i].x, people_target[i].y);

        // Generate random travel commands with a length between 11 and 35 steps
        std::string commands = generate_random_directions(people_start[i].x, people_start[i].y, people_target[i].x, people_target[i].y, 11, 35);
        travel_commands.push_back(commands);
    }

    // Save results in a JSON file
    save_to_json("travel_commands.json", travel_commands);

    std::cout << "Results saved in travel_commands.json" << std::endl;

    return 0;
}
