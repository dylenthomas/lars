#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char randomChar() { return 'a' + rand() % 26; }

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("There should only be one arg, number of letters.\n");
        return 0;
    }

    srandom(time(NULL));
   
    int num = atoi(argv[1]);
    for (int i = 0; i < 50; i++) {
        for (int i = 0; i < num; i++) {
            putchar(randomChar());
        }
        putchar('\n');
    }
}
