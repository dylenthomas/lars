#include <string.h>

#include "config_parser.h"

/*
 * JSON Parser for the keywords JSON config
 * Data structure is key : list in a single JSON object, for n keys
*/

// TODO: Switch to text file based parsing using new lines (I don't know why I wanted to use JSON)

// djb2 hash function from http://www.cse.yorku.ca/~oz/hash.html
uint64_t djb2Hash(char* str) {
    uint64_t hash = 5381;
    int c;

    while ((c = *str++)) {
        // equivalent: hash * 33 + c
        // except bit shifting to do so is just faster
        //  shifting 5 = 2^5 = 32, then + hash is 33
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

static void fileOpErrorCheck(int opperationReturn, FILE* fptr) {
    if (opperationReturn != 0) {
        printf("Error! %d\n", errno);
        fclose(fptr);
        exit(1);
    }
}

static int peekNext(FILE* fptr) {
    // Check what the next character is
    fpos_t starting_pos;
    (void)fileOpErrorCheck(fgetpos(fptr, &starting_pos), fptr);
  
    // move forward one byte
    (void)fileOpErrorCheck(fseek(fptr, 1, SEEK_CUR), fptr);

    int peeked = fgetc(fptr);
    (void)fileOpErrorCheck(fsetpos(fptr, &starting_pos), fptr);

    return peeked;
}

static int getKey(FILE* fptr, char key[], int longest_key) {
    // first check if where we are is a key
    // second if its a key check if its too long
    // third if not, then return the key.
    
    // Check if we are at a key
    int k = 0;
    int c;
    fpos_t starting_pos;
    (void)fileOpErrorCheck(fgetpos(fptr, &starting_pos), fptr);
    
    // if we hit a \n without finding a : then we are not at a key
    while ((c = fgetc(fptr)) != ':') {
        if (c == '\n' || c == -1) { 
            (void)fileOpErrorCheck(fsetpos(fptr, &starting_pos), fptr);
            return 0; 
        }
        if (c != '"' && c != ' ') { k++; }
    }
    if (k >= longest_key) {
        // we only get here if the while loop did more iterations than a key should be long
        // and we found a :, so there is key.
        printf("Found a key that is too long! Be sure to check all keys!\n");
        fclose(fptr);
        exit(1);
    }

    (void)fileOpErrorCheck(fsetpos(fptr, &starting_pos), fptr);

    // ensure the  key buffer is reset
    int i = 0;
    while (i < longest_key + 1) { key[i] = '\0'; i++; }

    // move until the start of the key
    while ((c = fgetc(fptr)) != '"')

    // we are back at the first ", but we know there is a key here that is the right length
    k = 0;
    do {
        key[k] = fgetc(fptr);
        k++; 
    } while (peekNext(fptr) != ':');

    // move the file pointer to after the key since we no longer need to be at the key
    while ((c = fgetc(fptr)) != '\n'); 
     
    return 1;
}

struct keywordHM createKeywordHM (const char* path) {
    int c;

    FILE *fptr;
    fptr = fopen(path, "r");
    (void)assert(fptr != NULL); 
    
    int longest_key = 2;
    char current_key[longest_key + 1] = {};

    int num_keys = 0;
    while ((c = fgetc(fptr)) != -1) {
        if (c == '\'') {
            printf("Remove single quotes from file!\n");
            fclose(fptr);
            exit(1);
        }
        // wait until we hit a new line
        while (c != '\n' && c != -1){
            c = fgetc(fptr);
            putchar(c); 
            if (c == '\'') {
                printf("Remove single quotes from file!\n");
                fclose(fptr);
                exit(1);
            }
        }
        // once we hit a new line check for a key
        num_keys += getKey(fptr, current_key, longest_key);
    }

    // set a hard limit on number of keys that shouldn't be exceeded
    long file_size = ftell(fptr);
    if (file_size == -1) {
        printf("Failed to get file size, %d!\n", errno);
        fclose(fptr);
        exit(1);
    }
    else if (num_keys > file_size) {
        printf(
            "Number of keys found is greater than the length of the file!\n File length: %ld, Number of Keys: %d\n",
            file_size, num_keys);
        fclose(fptr);
        exit(1);
    }

    printf("\nFound %d keys\n", num_keys);
    
    // go back to start of file so we can collect the data
    rewind(fptr);

    struct keywordHM hM;
    hM.len = num_keys;
    hM.items = calloc(num_keys, sizeof(struct item*));

    int key_ind;
    int buffer_len = 500; // shouldn't be anything larger than this
    char buffer[buffer_len] = {}; 
    while ((c = fgetc(fptr)) != -1) {
        if (c == '\'') {
            printf("Remove single quotes from file!\n");
            fclose(fptr);
            exit(1);
        }

        int i = 0;
        int started_word = 0;
        
        // wait until we hit a new line
        while (c != '\n' && c != -1) {
            c = fgetc(fptr);
            if (c == '\'') {
                printf("Remove single quotes from file!\n");
                fclose(fptr);
                exit(1);
            }    

            if (c == '"' && !started_word) { started_word = 1; continue; /* skip the " char  */}
            else if (c == '"' && started_word) {
                started_word = 0;

                /* have to force the struct to be assigned on the 
                  heap so it doesn't die once this stack is de-allocated 
                */
                struct item* _item = malloc(sizeof(struct item));
                _item->value = djb2Hash(buffer);

                _item->next = hM.items[key_ind];
                hM.items[key_ind] = _item;
               
                printf("%s\n", buffer);

                int _i = 0;
                while (_i < buffer_len) { buffer[_i] = '\0'; _i++; }

                i = 0;
            }

            if (i == buffer_len - 2) {
                printf("About to overflow keyword buffer (%d bytes)! There might be a keyword that is too long.\n", buffer_len);
                fclose(fptr);
                exit(1);
            }
            if (started_word) { buffer[i] = c; i++; }
        };
        
        // once we hit a new line check for a key
        if(getKey(fptr, current_key, longest_key) > 0) {
            uint64_t key_hash = djb2Hash(current_key);
            key_ind = key_hash % num_keys;
        }

    }

    fclose(fptr);
    return hM;
}
