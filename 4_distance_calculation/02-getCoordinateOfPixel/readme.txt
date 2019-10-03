#If the program doesn't compile successfully, use the following command to compile.

g++ main.cpp -o main `pkg-config --libs --cflags opencv`

#AFter compilation a binary file named "main" is generated.

# Run the program
./main <path/to/image/file>