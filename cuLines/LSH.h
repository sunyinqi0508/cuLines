#ifndef _CULINES_H
#define _CULINES_H

class HashTable;
class LshFunc;

void initialize(const char* filename);
void doCriticalPointQuery(const char *cp_filename);

extern float* alpha;
#endif
