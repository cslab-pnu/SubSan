#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("arg error\n");
        return -1;
    }

    int idx = -(atoi(argv[1]));
    char *ptr = (char*) malloc(64);

    printf("obj: %p\n", ptr);
    printf("obj[%d] access...\n", idx);
    printf("%x\n", ptr[idx]);

    free(ptr);
    return 0;
}