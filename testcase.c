#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_PEOPLE 5

// Function to generate grid data
void generate_grid(int grid_width, int grid_height, const char* filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    // Generate random grid data between 0 and 100
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_width; j++) {
            fprintf(file, "%d ", rand() % 100);  // Random values between 0-99
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Function to generate people data (start and target positions)
void generate_people(int grid_width, int grid_height, int num_people, const char* filename, const char* directions_filename) {
    FILE *file = fopen(filename, "w");
    FILE *directions_file = fopen(directions_filename, "w");
    if (!file || !directions_file) {
        printf("Error opening file: %s or %s\n", filename, directions_filename);
        exit(1);
    }

    // Generate random people positions (start and target positions)
    for (int i = 0; i < num_people; i++) {
        int start_x = rand() % grid_width;
        int start_y = rand() % grid_height;
        int target_x, target_y;

        // Ensure minimum distance of 1/4th grid size between start and target
        do {
            target_x = rand() % grid_width;
            target_y = rand() % grid_height;
        } while (abs(target_x - start_x) < grid_width / 4 || abs(target_y - start_y) < grid_height / 4);

        fprintf(file, "%d %d %d %d\n", start_x, start_y, target_x, target_y);

        // Directly generate directions for each person
        int x_diff = target_x - start_x;
        int y_diff = target_y - start_y;

        fprintf(directions_file, "\"person%d\": \"", i + 1);  // Include the person index for clarity

        // Calculate and print the horizontal direction (left or right)
        while (x_diff != 0) {
            if (x_diff > 0) {
                fprintf(directions_file, "right, ");
                x_diff--;
            } else {
                fprintf(directions_file, "left, ");
                x_diff++;
            }
        }

        // Calculate and print the vertical direction (up or down)
        while (y_diff != 0) {
            if (y_diff > 0) {
                fprintf(directions_file, "up, ");
                y_diff--;
            } else {
                fprintf(directions_file, "down, ");
                y_diff++;
            }
        }

        // Remove the trailing comma and space, then close the quote for the person
        fseek(directions_file, -2, SEEK_CUR);  // Go back 2 bytes to remove last comma
        fprintf(directions_file, "\"\n");
    }

    fclose(file);
    fclose(directions_file);
}

int main() {
    srand(time(NULL));

    // User inputs for grid size, number of people, and number of obstacles
    int grid_width, grid_height, num_people;

    printf("Enter grid width: ");
    scanf("%d", &grid_width);

    printf("Enter grid height: ");
    scanf("%d", &grid_height);

    printf("Enter number of people: ");
    scanf("%d", &num_people);

    if (num_people > MAX_PEOPLE) {
        printf("Warning: You have entered a large number of people (%d). Consider reducing this number.\n", num_people);
    }

    // Generate grid and people data
    generate_grid(grid_width, grid_height, "grid.txt");
    generate_people(grid_width, grid_height, num_people, "people.txt", "directions.json");

    printf("Test case generation completed. Files: grid.txt, people.txt, directions.json\n");

    return 0;
}
