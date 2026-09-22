#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    char *ptr = (char*) malloc(64);
    printf("obj: %p\n", ptr);

    free(ptr);
    printf("obj %p deallocated...\n", ptr);

    printf("use-after-free!!!\n");
    printf("%c\n", ptr[argc]);

    return 0;
}