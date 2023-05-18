/*
   Copyright 2021, University of Southern California

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/


#ifndef MESSAGE_H
#define MESSAGE_H
#include"types.h"
#define LISTEN_TAG 0
#define ANSWER_TAG 1
#define ASK_IN 0
#define ASK_OUT 1
#define EXIT 2
#define LISTEN_TAG_NON 2
#define ANSWER_TAG_NON 3
#define ASK_FEATURE 0
#define ASK_LABEL 1

struct Edge_Query
{
    uint8_t type;
    VertexId v;
} __attribute__((packed));
struct Vertex_Query
{
    uint8_t type;
    VertexId v;
} __attribute__((packed));




#endif
