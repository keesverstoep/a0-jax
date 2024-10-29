#
# Copyright 2024 Vrije Universiteit Amsterdam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
# import goedels
# import binomium
from . import goedels
from . import binomium

awari_path = os.getenv("AWARI_PATH", ".") 

# Based on the original version written in C.
#
# This implements the coding used that omits unreachable positions,
# i.e., the opponent always has at least one empty pit (except for
# the starting position) since it in the previous turn it must
# have emptied one

# Expand the goedel array to full 12,64,64] dimensions
def full_goedels():
    goedels_expanded = [[[0 for z in range(64)] for y in range(64)] for x in range(12)]
    for x in range(len(goedels.goedels)):
        for y in range(len(goedels.goedels[x])):
            for z in range(len(goedels.goedels[x][y])):
                goedels_expanded[x][y][z] = goedels.goedels[x][y][z]
    return goedels_expanded 
    
goedel_arr = full_goedels()

# class Board():
class db_board():
    def __init__(self, pit0, pit1, pit2, pit3, pit4, pit5, pit6, pit7, pit8, pit9, pit10, pit11):
        self.pits = [pit0, pit1, pit2, pit3, pit4, pit5, pit6, pit7, pit8, pit9, pit10, pit11]

    @classmethod
    def goedel_init(cls, goedelNumber, nrSeeds):
        pits = [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0]
        pit = 11;
        index = nrSeeds - 5

        while True:
            pits[pit] = 0
            while nrSeeds > 0:
                newGoedel = goedelNumber - binomium.binomium[pit][nrSeeds + 8]
                if pits[pit] > 0:
                    newGoedel += binomium.binomium[pit][index + 8]
                if newGoedel < 0:
                    break
                goedelNumber = newGoedel

                nrSeeds -= 1
                pits[pit] += 1
                index -= 1

            index += 1
            if pits[pit] == 0:
                pit -= 1
                break
            pit -= 1

        while True:
            seeds = 0
            goedelsPtr = goedel_arr[pit][nrSeeds + 8]

            while nrSeeds > seeds and goedelNumber >= goedelsPtr[seeds + 1]:
                seeds += 1

            goedelNumber -= goedelsPtr[seeds]
            nrSeeds -= seeds
            pits[pit] = seeds
            pit -= 1
            if pit < 0:
                break

        return cls(pits[0], pits[1], pits[2], pits[3], pits[4], pits[5],
                   pits[6], pits[7], pits[8], pits[9], pits[10], pits[11])

    def count_seeds(self):
        sum = 0
        for i in range(12):
            sum += self.pits[i]
        return sum

    def GoedelNumber(self, nrSeeds):
        goedelNumber = 0
        seeds = 0
        pit = 11

        # NOTE: this code makes use of the fact that the opponent (except from the
        # starting position) always has at least one pit empty because of a previous move
        while self.pits[pit] > 0:
            seeds = self.pits[pit]
            goedelNumber += goedel_arr[pit][nrSeeds + 8][seeds]
            goedelNumber -= goedel_arr[pit][nrSeeds - pit + 5 + 8][seeds - 1]
            nrSeeds -= seeds;
            pit -= 1
            if pit < 0:
                # should just happen for the starting position, TODO: check
                print("GoedelNumber should only be called for positions where opponent has moved")
                return -1

        while nrSeeds > 0:
            seeds = self.pits[pit]
            goedelNumber += goedel_arr[pit][nrSeeds + 8][seeds]
            pit -= 1
            nrSeeds -= seeds

        return goedelNumber

    def nrStates(self, nrSeeds):
        return goedel_arr[11][nrSeeds + 8][6]

    def score(self, goedelNumber, nrSeeds):
        filename = awari_path + "/scores.%d" % nrSeeds
        f = open(filename, 'rb')
        f.seek(goedelNumber)
        byte = f.read(1)
        f.close()
        result = int.from_bytes(byte, byteorder="little", signed=True) 
        return result

    def __repr__(self):
        str = "+----+----+----+----+----+----+\n|"

        pit = 12
        while pit > 6:
            pit -= 1
            str += "%3d |" % self.pits[pit]

        str += "\n+----+----+----+----+----+----+\n|"

        pit = 0
        while pit < 6:
            str += "%3d |" % self.pits[pit]
            pit += 1

        str += "\n+----+----+----+----+----+----+"
        return str

def test_board(pits):
    # b =  Board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
    b =  db_board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
	       pits[6],pits[7],pits[8],pits[9],pits[10],pits[11])
    print(b)
    seeds = b.count_seeds()
    g = b.GoedelNumber(seeds)
    print("seeds", seeds, "goedel", g)

    # new_b = Board.goedel_init(g, seeds)
    new_b = db_board.goedel_init(g, seeds)
    print("new board:\n" + str(new_b))
    print()

def test_position(pits):
    # b =  Board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
    b =  db_board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
	       pits[6],pits[7],pits[8],pits[9],pits[10],pits[11])
    print(b)
    seeds = b.count_seeds()
    g = b.GoedelNumber(seeds)
    score = b.score(g, seeds)
    print("nrStates", b.nrStates(seeds)) 
    print("score:", score)

if __name__ == "__main__":
    test_board([0,2,3,4,5,6,  0,2,3,4,5,6])
    test_board([0,9,3,13,2,0, 0,5,4,4,4,4])
    test_board([0,1,2,0,0,0,  0,1,2,0,0,0])
    test_board([0,1,2,0,0,0,  1,0,2,0,0,0])
    test_board([4,4,4,4,4,4,  0,5,5,5,5,4])
    test_board([4,4,4,4,4,4,  4,0,5,5,5,5])
    test_board([5,4,4,4,4,4,  4,4,0,5,5,5])
    test_board([5,5,4,4,4,4,  4,4,4,0,5,5])
    test_board([5,5,5,4,4,4,  4,4,4,4,0,5])
    test_board([5,5,5,5,4,4,  4,4,4,4,4,0])

    test_position([0,1,2,0,0,0,  0,1,2,0,0,0])
    test_position([0,1,2,0,0,0,  1,0,2,0,0,0])

    # generate and test all 6-seed boards
    # b = Board(6,0,0,0,0,0,  0,0,0,0,0,0)
    b = db_board(6,0,0,0,0,0,  0,0,0,0,0,0)
    seeds = b.count_seeds()
    nrstates = b.nrStates(seeds)
    print("db for %d seeds has %d entries:" % (seeds, nrstates))
    for i in range(b.nrStates(b.count_seeds())):
        # bi = Board.goedel_init(i, seeds)
        bi = db_board.goedel_init(i, seeds)
        print("index = %d, score = %d" % (i, bi.score(i, seeds)))
        print(str(bi))
