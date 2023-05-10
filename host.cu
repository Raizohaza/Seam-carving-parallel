#include <stdio.h>
#include <stdint.h>

using namespace std;


int WIDTH;
__device__ int d_WIDTH;

int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
__constant__ int d_xSobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ int d_ySobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
const int filterWidth = 3;

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}
float computeError(uchar3 * a1, uchar3 * a2, int n) {
    float err = 0;
    for (int i = 0; i < n; i++) {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}


void printError(char * msg, uchar3 * in1, uchar3 * in2, int width, int height) {
	float err = computeError(in1, in2, width * height);
	printf("%s: %f\n", msg, err);
}

void printDeviceInfo() {
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("_____________GPU info_____________\n");
    printf("|Name:                   %s|\n", devProv.name);
    printf("|Compute capability:          %d.%d|\n", devProv.major, devProv.minor);
    printf("|Num SMs:                      %d|\n", devProv.multiProcessorCount);
    printf("|Max num threads per SM:     %d|\n", devProv.maxThreadsPerMultiProcessor); 
    printf("|Max num warps per SM:         %d|\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("|GMEM:           %zu byte|\n", devProv.totalGlobalMem);
    printf("|SMEM per SM:          %zu byte|\n", devProv.sharedMemPerMultiprocessor);
    printf("|SMEM per block:       %zu byte|\n", devProv.sharedMemPerBlock);
    printf("|________________________________|\n");
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
	 void printTime(char * s) {
        printf("Processing time of %s: %f ms\n\n", s, Elapsed());
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
    FILE * f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);
    
    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);
    
    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName)
{
    FILE * f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }   

    fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * originalWidth + c;
            fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
        }
    }
    
    fclose(f);
}

void convertRgb2Gray_host(uchar3 * rgbPic, int width, int height, uint8_t * grayPic) {
    for (int r = 0; r < height; ++r) 
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            grayPic[i] = 0.299f * rgbPic[i].x + 0.587f * rgbPic[i].y + 0.114f * rgbPic[i].z;
        }
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += abs((int)a1[i] - (int)a2[i]);
	err /= n;
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

/**
 * @param argc[1] name of the input file (.pmn)
 * @param argc[2] name of output file with no extension, created by using host & device
 * @param argc[3] horizontal of image you want to resize 
 * @param argc[4] - optional - default(32): blocksize.x
 * @param argc[5] - optional - default(32): blocksize.y
 */
void checkInput(int argc, char ** argv, int &width, int &height, uchar3 * &rgbPic, int &desiredWidth, dim3 &blockSize) {
    if (argc != 4 && argc != 6) {
        printf("The number of arguments is invalid\n");
        exit(EXIT_FAILURE);
    }

    // Read file
    readPnm(argv[1], width, height, rgbPic);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    WIDTH = width;
    CHECK(cudaMemcpyToSymbol(d_WIDTH, &width, sizeof(int)));

    // Check user's desired width
    desiredWidth = atoi(argv[3]);

    if (desiredWidth <= 0 || desiredWidth >= width) {
        printf("Your desired width must between 0 & current picture's width!\n");
        exit(EXIT_FAILURE);
    }

    // Block size
    if (argc == 6) {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    } 

    // Check GPU is working or not
    printDeviceInfo();
}

// HOST
int getPixelEnergy(uint8_t * grayPixels, int row, int col, int width, int height) {
    int x_kernel = 0;
    int y_kernel = 0;

    for (int i = 0; i < 3; ++i) { // 3: filter width
        for (int j = 0; j < 3; ++j) {
            int r = min(max(0, row - 1 + i), height - 1); // 0 <= row - 1 + i < height
            int c = min(max(0, col - 1 + j), width - 1); // 0 <= col - 1 + j < width

            uint8_t pixelVal = grayPixels[r * WIDTH + c];//

            x_kernel += pixelVal * xSobel[i][j];// Convolution with x-Sobel
            y_kernel += pixelVal * ySobel[i][j];// Convolution with y-Sobel
        }
    }
    return abs(x_kernel) + abs(y_kernel);// Add matrix
}

void energyToTheEnd(int * energy, int * minimalEnergy, int width, int height) {
    for (int c = 0; c < width; ++c) {
        minimalEnergy[c] = energy[c];
    }
    for (int r = 1; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * WIDTH + c;
            int aboveIdx = (r - 1) * WIDTH + c;

            int min = minimalEnergy[aboveIdx];
            if (c > 0 && minimalEnergy[aboveIdx - 1] < min) {
                min = minimalEnergy[aboveIdx - 1];
            }
            if (c < width - 1 && minimalEnergy[aboveIdx + 1] < min) {
                min = minimalEnergy[aboveIdx + 1];
            }

            minimalEnergy[idx] = min + energy[idx];
        }
    }
}

void hostResizing(uchar3 * inPixels, int width, int height, int desiredWidth, uchar3 * outPixels) {
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    // Allocating memory
    int * energy = (int *)malloc(width * height * sizeof(int));
    int * minimalEnergy = (int *)malloc(width * height * sizeof(int));
    
    // Get grayscale
    uint8_t * grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convertRgb2Gray_host(inPixels, width, height, grayPixels);

    // Calculate all pixels energy
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            energy[r * WIDTH + c] = getPixelEnergy(grayPixels, r, c, width, height);
        }
    }

    while (width > desiredWidth) {
        // Calculate energy to the end. (go from bottom to top)
        energyToTheEnd(energy, minimalEnergy, width, height);

        // find min index of last row
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c) {
            if (minimalEnergy[r * WIDTH + c] < minimalEnergy[r * WIDTH + minCol])
                minCol = c;
        }

        // Find and remove seam from last to first row
        for (; r >= 0; --r) {
            // remove seam pixel on row r
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * WIDTH + i] = outPixels[r * WIDTH + i + 1];
                grayPixels[r * WIDTH + i] = grayPixels[r * WIDTH + i + 1];
                energy[r * WIDTH + i] = energy[r * WIDTH + i + 1];
            }

            // Update energy
            if (r < height - 1) {
                int affectedCol = max(0, prevMinCol - 2);

                while (affectedCol <= prevMinCol + 2 && affectedCol < width - 1) {
                    energy[(r + 1) * WIDTH + affectedCol] = getPixelEnergy(grayPixels, r + 1, affectedCol, width - 1, height);
                    affectedCol += 1;
                }
            }

            // find to the top
            if (r > 0) {
                prevMinCol = minCol;

                int aboveIdx = (r - 1) * WIDTH + minCol;
                int min = minimalEnergy[aboveIdx], minColCpy = minCol;
                if (minColCpy > 0 && minimalEnergy[aboveIdx - 1] < min) {
                    min = minimalEnergy[aboveIdx - 1];
                    minCol = minColCpy - 1;
                }
                if (minColCpy < width - 1 && minimalEnergy[aboveIdx + 1] < min) {
                    minCol = minColCpy + 1;
                }
            }
        }

        int affectedCol;
        for (affectedCol=max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol) {
            energy[affectedCol] = getPixelEnergy(grayPixels, 0, affectedCol, width - 1, height);
        }

        --width;
    }
    
    free(grayPixels);
    free(minimalEnergy);
    free(energy);

    timer.Stop();
    timer.printTime((char *)"host");
}

int main(int argc, char ** argv) {   

    int width, height, desiredWidth;
    uchar3 * rgbPic;
    dim3 blockSize(32, 32);

    // Check user's input
    checkInput(argc, argv, width, height, rgbPic, desiredWidth, blockSize);

    // HOST
    uchar3 * out_host = (uchar3 *)malloc(width * height * sizeof(uchar3));
    hostResizing(rgbPic, width, height, desiredWidth, out_host);

    writePnm(out_host, desiredWidth, height, width, concatStr(argv[2], "_host.pnm"));

    // Free memories
    free(rgbPic);
    free(out_host);
    free(out_device);
}
