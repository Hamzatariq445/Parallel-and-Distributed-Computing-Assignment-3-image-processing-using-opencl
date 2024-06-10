#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>

int main() {
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input_mem, output_mem;
    cl_int err;

    // Load the color image data using OpenCV
    cv::Mat input_image = cv::imread("ISIC_3610466.jpg", cv::IMREAD_COLOR);
    if (input_image.empty()) {
        printf("Error loading image file.\n");
        return 1;
    }

    int image_width = input_image.cols;
    int image_height = input_image.rows;

    // Get the first platform and device
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting platform or device.");
        return 1;
    }

    // Create a context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating context.");
        return 1;
    }

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating command queue.");
        return 1;
    }

    // Declare source as pointer to array of strings
const char *source[] = {
    "_kernel void grayscale(_global const uchar* input, __global uchar* output, int width, int height) {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "    if (x < width && y < height) {\n"
    "        int index = y * width * 3 + x * 3;\n"
    "        uchar r = input[index + 0];\n"
    "        uchar g = input[index + 1];\n"
    "        uchar b = input[index + 2];\n"
    "        uchar gray = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);\n"
    "        int output_index = y * width + x;\n"
    "        output[output_index] = gray;\n"
    "    }\n"
    "}"
};

// Create the program
program = clCreateProgramWithSource(context, 1, source, NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error creating program.");
        return 1;
    }

    // Build the program
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error building program.");
        return 1;
    }

    // Create the kernel
    kernel = clCreateKernel(program, "grayscale", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating kernel.");
        return 1;
    }

    // Create memory buffers
    output_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_width * image_height * sizeof(unsigned char), NULL, &err);
    input_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_width * image_height * 3 * sizeof(unsigned char), input_image.data, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating memory buffers.");
        return 1;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_mem);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_mem);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &image_width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &image_height);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arguments.");
        return 1;
    }

    // Execute the kernel
    size_t global_size[2] = { image_width, image_height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error executing kernel.");
        return 1;
    }

    // Read the memory buffer (grayscale image data)
    unsigned char output_image[image_width * image_height];
    err = clEnqueueReadBuffer(queue, output_mem, CL_TRUE, 0, image_width * image_height * sizeof(unsigned char), output_image, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading memory buffer.\n");
        return 1;
    }

    // Save the grayscale image data using OpenCV
    cv::Mat output_mat(image_height, image_width, CV_8UC1, output_image);
    cv::imwrite("converted_grayscale.png", output_mat);

    // Clean up
    if (input_mem) clReleaseMemObject(input_mem);
    if (output_mem) clReleaseMemObject(output_mem);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);

    return 0;
}
